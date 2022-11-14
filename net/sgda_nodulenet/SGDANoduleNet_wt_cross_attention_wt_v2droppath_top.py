import sys

from net.uni_layer import *

from configs.sgda_config import config
from configs.sgda_config import config as cfg
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm
from net.sgda.domain_attention_module_wt_cross_attention_wt_droppath import DomainAttention as DomainAttention_wt_cross_attention_wt_droppath
from torch.cuda.amp import autocast as autocast


bn_momentum = 0.1
affine = True


class ResBlock3d(nn.Module):
  def __init__(self, n_in, n_out, stride=1):
    super(ResBlock3d, self).__init__()
    self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

    if stride != 1 or n_out != n_in:
      self.shortcut = nn.Sequential(
        nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
        nn.BatchNorm3d(n_out, momentum=bn_momentum))
    else:
      self.shortcut = None

  def forward(self, x):
    residual = x
    if self.shortcut is not None:
      residual = self.shortcut(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    out += residual
    out = self.relu(out)
    return out

class DAResBlock3d_wt_cross_attention_wt_droppath(nn.Module):
    def __init__(self, n_in, n_out, net_mode, drop_path, stride=1, fixed_block=False, reduction=16):
        super(DAResBlock3d_wt_cross_attention_wt_droppath, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        if cfg['use_mux']:
          self.bn1 = nn.ModuleList([nn.BatchNorm3d(n_out, momentum=bn_momentum) for datasets in cfg['num_class']])
        else:
          self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        if cfg['use_mux']:
          self.bn2 = nn.ModuleList([nn.BatchNorm3d(n_out, momentum=bn_momentum) for datasets in cfg['num_class']])
        else:
          self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.domain_attention = DomainAttention_wt_cross_attention_wt_droppath(n_out, reduction=16, nclass_list=cfg['num_class'], fixed_block=fixed_block,
                                                groups=config['groups'], mode='noG_yesD', attention_dir=True, drop_path=drop_path, net_mode=net_mode)

        if stride != 1 or n_out != n_in:
            if cfg['use_mux']:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                    BnMux(n_out))
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                    nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x, cls_ind):
        residual = x
        if self.shortcut is not None:
          if cfg['use_mux']:
            residual = self.shortcut(x, cls_ind)
          else:
            residual = self.shortcut(x)
        out = self.conv1(x)
        if cfg['use_mux']:
          out = self.bn1[cls_ind](out)
        else:
          out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if cfg['use_mux']:
          out = self.bn2[cls_ind](out)
        else:
          out = self.bn2(out)
        out = self.domain_attention(out)

        out += residual
        out = self.relu(out)
        return out

class BnMux(nn.Module):
    def __init__(self, planes):
        super(BnMux, self).__init__()
        self.bn = nn.ModuleList(
            [nn.BatchNorm3d(planes, momentum=bn_momentum) for n_classes in cfg['num_class']]
        )

    def forward(self, x, cls_ind):
        out = self.bn[cls_ind](x)
        return out


class DAFeatureNet(nn.Module):
  def __init__(self, config, in_channels, out_channels, net_mode):
    super(DAFeatureNet, self).__init__()
    self.preBlock = nn.Sequential(
      nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, stride=2),
      nn.BatchNorm3d(24, momentum=bn_momentum),
      nn.ReLU(inplace=True),
      nn.Conv3d(24, 24, kernel_size=3, padding=1),
      nn.BatchNorm3d(24, momentum=bn_momentum),
      nn.ReLU(inplace=True))
    dpr = [x.item() for x in torch.linspace(0, config['drop_path'], 6)]

    fixed_block = False
    self.forw1_1 = ResBlock3d(24, 32)
    self.forw1_2 = ResBlock3d(32, 32)

    self.forw2_1 = ResBlock3d(32, 64)
    self.forw2_2 = ResBlock3d(64, 64)

    self.forw3_1 = ResBlock3d(64, 64)
    self.forw3_2 = ResBlock3d(64, 64)
    self.forw3_3 = ResBlock3d(64, 64)

    self.forw4_1 = ResBlock3d(64, 64)
    self.forw4_2 = ResBlock3d(64, 64)
    self.forw4_3 = ResBlock3d(64, 64)

    # skip connection in U-net
    self.back2_1 = DAResBlock3d_wt_cross_attention_wt_droppath(128, 128, net_mode, drop_path=dpr[3], fixed_block=fixed_block)
    self.back2_2 = DAResBlock3d_wt_cross_attention_wt_droppath(128, 128, net_mode, drop_path=dpr[4], fixed_block=fixed_block)
    self.back2_3 = DAResBlock3d_wt_cross_attention_wt_droppath(128, 128, net_mode, drop_path=dpr[5], fixed_block=fixed_block)
      # 64 + 64 + 3, where 3 is the channeld dimension of coord

    # skip connection in U-net
    self.back3_1 = DAResBlock3d_wt_cross_attention_wt_droppath(128, 64, net_mode, drop_path=dpr[0], fixed_block=fixed_block)
    self.back3_2 = DAResBlock3d_wt_cross_attention_wt_droppath(64, 64, net_mode, drop_path=dpr[1], fixed_block=fixed_block)
    self.back3_3 = DAResBlock3d_wt_cross_attention_wt_droppath(64, 64, net_mode, drop_path=dpr[2], fixed_block=fixed_block)

    self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                 return_indices=True)
    self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                 return_indices=True)
    self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                 return_indices=True)
    self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                 return_indices=True)

    # upsampling in U-net
    self.path1 = nn.Sequential(
      nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
      nn.BatchNorm3d(64),
      nn.ReLU(inplace=True))

    # upsampling in U-net
    self.path2 = nn.Sequential(
      nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
      nn.BatchNorm3d(64),
      nn.ReLU(inplace=True))

  def forward(self, x, cls_ind):
    out = self.preBlock(x)  # 16
    out_pool = out
    out1 = self.forw1_1(out_pool, cls_ind)
    out1 = self.forw1_2(out1, cls_ind)# 32
    out1_pool, _ = self.maxpool2(out1)
    out2 = self.forw2_1(out1_pool, cls_ind)
    out2 = self.forw2_2(out2, cls_ind)# 64
    # out2 = self.drop(out2)
    out2_pool, _ = self.maxpool3(out2)
    out3 = self.forw3_1(out2_pool, cls_ind)
    out3 = self.forw3_2(out3, cls_ind)
    out3 = self.forw3_3(out3, cls_ind)# 96
    out3_pool, _ = self.maxpool4(out3)
    out4 = self.forw4_1(out3_pool, cls_ind)
    out4 = self.forw4_2(out4, cls_ind)
    out4 = self.forw4_3(out4, cls_ind)# 96
    # out4 = self.drop(out4)

    rev3 = self.path1(out4)
    comb3 = self.back3_1(torch.cat((rev3, out3), 1), cls_ind)
    comb3 = self.back3_2(comb3, cls_ind)
    comb3 = self.back3_3(comb3, cls_ind)# 96+96
    rev2 = self.path2(comb3)
    comb2 = self.back2_1(torch.cat((rev2, out2), 1), cls_ind)
    comb2 = self.back2_2(comb2, cls_ind)
    comb2 = self.back2_3(comb2, cls_ind)# 64+64

    return [x, out1, comb2], out2


class RpnHead(nn.Module):
  def __init__(self, config, in_channels=128):
    super(RpnHead, self).__init__()
    self.drop = nn.Dropout3d(p=0.5, inplace=False)
    # self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1), nn.ReLU())
    self.conv = nn.ModuleList([nn.Conv3d(in_channels, 64, kernel_size=1) for i in range(config['num_datasets'])])
    self.relu = nn.ReLU()
    self.logits = nn.ModuleList(
      [nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1) for i in range(config['num_datasets'])])
    self.deltas = nn.ModuleList(
      [nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1) for i in range(config['num_datasets'])])

  def forward(self, f, cls_ind):
    # out = self.drop(f)
    if config['rpn_univ']:
      cls_ind = 0
    else:
      cls_ind = cls_ind

    # out = self.conv(f)
    out = self.conv[cls_ind](f)
    out = self.relu(out)

    logits = self.logits[cls_ind](out)
    deltas = self.deltas[cls_ind](out)
    size = logits.size()
    logits = logits.view(logits.size(0), logits.size(1), -1)
    logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)

    size = deltas.size()
    deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
    deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)

    return logits, deltas


class RcnnHead(nn.Module):
  def __init__(self, config, in_channels=128):
    super(RcnnHead, self).__init__()
    self.num_class = config['num_class']
    self.crop_size = config['rcnn_crop_size']

    self.fc1 = nn.ModuleList(
      [nn.Linear(in_channels * self.crop_size[i][0] * self.crop_size[i][1] * self.crop_size[i][2], 512) \
       for i in range(config['num_datasets'])])
    self.fc2 = nn.ModuleList([nn.Linear(512, 256) for i in range(config['num_datasets'])])
    self.logit = nn.ModuleList([nn.Linear(256, self.num_class[i]) for i in range(config['num_datasets'])])
    self.delta = nn.ModuleList([nn.Linear(256, self.num_class[i] * 6) for i in range(config['num_datasets'])])

  def forward(self, crops, cls_ind):
    x = crops.view(crops.size(0), -1)
    x = F.relu(self.fc1[cls_ind](x), inplace=True)
    x = F.relu(self.fc2[cls_ind](x), inplace=True)
    # x = F.dropout(x, 0.5, training=self.training)
    logits = self.logit[cls_ind](x)
    deltas = self.delta[cls_ind](x)

    return logits, deltas


def crop_mask_regions(masks, crop_boxes):
  out = []
  for i in range(len(crop_boxes)):
    b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
    m = masks[i][z_start:z_end, y_start:y_end, x_start:x_end].contiguous()
    out.append(m)

  return out


def top1pred(boxes):
  res = []
  pred_cats = np.unique(boxes[:, -1])
  for cat in pred_cats:
    preds = boxes[boxes[:, -1] == cat]
    res.append(preds[0])

  res = np.array(res)
  return res


def random1pred(boxes):
  res = []
  pred_cats = np.unique(boxes[:, -1])
  for cat in pred_cats:
    preds = boxes[boxes[:, -1] == cat]
    idx = random.sample(range(len(preds)), 1)[0]
    res.append(preds[idx])

  res = np.array(res)
  return res


class CropRoi(nn.Module):
  def __init__(self, cfg, rcnn_crop_size):
    super(CropRoi, self).__init__()
    self.cfg = cfg
    self.rcnn_crop_size = rcnn_crop_size
    self.scale = cfg['stride']
    self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size']
    # self.crop = nn.ModuleList([F.adaptive_max_pool3d(self.rcnn_crop_size) for i in range(config['num_datasets'])])
    self.crop = nn.ModuleList([nn.AdaptiveMaxPool3d(self.rcnn_crop_size[i]) for i in range(config['num_datasets'])])

  def forward(self, f, inputs, proposals, cls_ind):
    self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

    crops = []
    for p in proposals:
      b = int(p[0])
      center = p[2:5]
      side_length = p[5:8]
      c0 = center - side_length / 2  # left bottom corner
      c1 = c0 + side_length  # right upper corner
      c0 = (c0 / self.scale).floor().long()
      c1 = (c1 / self.scale).ceil().long()
      minimum = torch.LongTensor([[0, 0, 0]]).cuda()
      maximum = torch.LongTensor(
        np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale).cuda()

      c0 = torch.cat((c0.unsqueeze(0), minimum), 0)
      c1 = torch.cat((c1.unsqueeze(0), maximum), 0)
      c0, _ = torch.max(c0, 0)
      c1, _ = torch.min(c1, 0)

      # Slice 0 dim, should never happen
      if np.any((c1 - c0).cpu().data.numpy() < 1):
        print(p)
        print('c0:', c0, ', c1:', c1)
      crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
      crop = self.crop[cls_ind](crop)
      crops.append(crop)

    crops = torch.stack(crops)

    return crops


class SGDANoduleNet_wt_cross_attention_wt_v2droppath_bottom(nn.Module):
  def __init__(self, cfg, mode='train'):
    super(SGDANoduleNet_wt_cross_attention_wt_v2droppath_bottom, self).__init__()

    self.cfg = cfg
    self.mode = mode
    self.feature_net = DAFeatureNet(cfg, 1, 128, net_mode=self.mode)
    self.rpn = RpnHead(cfg, in_channels=128)
    self.rcnn_head = RcnnHead(cfg, in_channels=64)
    self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
    # self.mask_head = MaskHead(config, in_channels=128)
    self.use_rcnn = False

    # self.rpn_loss = Loss(cfg['num_hard'])

  # @autocast()
  def forward(self, inputs, cls_ind, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
    # features, feat_4 = data_parallel(self.feature_net, (inputs)); #print('fs[-1] ', fs[-1].shape)
    features, feat_4 = self.feature_net(inputs, cls_ind)
    fs = features[-1]

    # self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs, cls_ind=cls_ind)
    rpn_logits_flat, rpn_deltas_flat = self.rpn(fs, cls_ind=cls_ind)

    b, D, H, W, _, num_class = rpn_logits_flat.shape

    rpn_logits_flat = rpn_logits_flat.view(b, -1, 1);  # print('rpn_logit ', self.rpn_logits_flat.shape)
    rpn_deltas_flat = rpn_deltas_flat.view(b, -1, 6);  # print('rpn_delta ', self.rpn_deltas_flat.shape)

    rpn_window = make_rpn_windows(fs, self.cfg)
    rpn_proposals = []
    if self.use_rcnn or self.mode in ['eval', 'test']:
      rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, rpn_window,
                              rpn_logits_flat, rpn_deltas_flat, cls_ind)
      # print 'length of rpn proposals', self.rpn_proposals.shape

    if self.mode in ['train', 'valid']:
      # self.rpn_proposals = torch.zeros((0, 8)).cuda()
      rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_target_weights = \
        make_rpn_target(self.cfg, self.mode, inputs, rpn_window, truth_boxes, truth_labels, cls_ind)

      if self.use_rcnn:
        # self.rpn_proposals = torch.zeros((0, 8)).cuda()
        rpn_proposals, rcnn_labels, rcnn_assigns, rcnn_targets = \
          make_rcnn_target(self.cfg, self.mode, inputs, rpn_proposals,
                           truth_boxes, truth_labels, cls_ind=cls_ind)

    # rcnn proposals
    detections = copy.deepcopy(rpn_proposals)
    ensemble_proposals = copy.deepcopy(rpn_proposals)

    if self.use_rcnn:
      if len(rpn_proposals) > 0:
        rcnn_crops = self.rcnn_crop(feat_4, inputs, rpn_proposals, cls_ind=cls_ind)
        # self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops, cls_ind=cls_ind)
        rcnn_logits, rcnn_deltas = self.rcnn_head(rcnn_crops, cls_ind=cls_ind)
        detections, keeps = rcnn_nms(self.cfg, self.mode, inputs, rpn_proposals, rcnn_logits, rcnn_deltas,
                                     cls_ind=cls_ind)

      if self.mode in ['eval']:
        # Ensemble
        fpr_res = get_probability(self.cfg, self.mode, inputs, rpn_proposals, rcnn_logits, rcnn_deltas,
                                  cls_ind=cls_ind)
        ensemble_proposals[:, 1] = (ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2

        return rpn_proposals, detections, ensemble_proposals

      return rpn_logits_flat, rpn_deltas_flat, rpn_labels, rpn_label_weights, rpn_targets, rpn_target_weights, \
             rcnn_logits, rcnn_deltas, rcnn_labels, rcnn_targets

    return rpn_logits_flat, rpn_deltas_flat, rpn_labels, rpn_label_weights, rpn_targets, rpn_target_weights

  # @autocast()
  def loss(self, rpn_logits_flat, rpn_deltas_flat, rpn_labels,
           rpn_label_weights, rpn_targets, rpn_target_weights,
           rcnn_logits=None, rcnn_deltas=None, rcnn_labels=None, rcnn_targets=None):
    cfg = self.cfg

    rcnn_cls_loss, rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
    rcnn_stats = None

    rpn_cls_loss, rpn_reg_loss, rpn_stats = \
      rpn_loss(rpn_logits_flat, rpn_deltas_flat, rpn_labels,
               rpn_label_weights, rpn_targets, rpn_target_weights, self.cfg, mode=self.mode)

    if self.use_rcnn:
      rcnn_cls_loss, rcnn_reg_loss, rcnn_stats = \
        rcnn_loss(rcnn_logits, rcnn_deltas, rcnn_labels, rcnn_targets)

    total_loss = rpn_cls_loss + rpn_reg_loss \
                 + rcnn_cls_loss + rcnn_reg_loss

    return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, total_loss, rpn_stats, rcnn_stats

  def set_mode(self, mode):
    assert mode in ['train', 'valid', 'eval', 'test']
    self.mode = mode
    if mode in ['train']:
      self.train()
    else:
      self.eval()

  def set_anchor_params(self, anchor_ids, anchor_params):
    self.anchor_ids = anchor_ids
    self.anchor_params = anchor_params

  def crf(self, detections, cls_ind):
    """
    detections: numpy array of detection results [b, z, y, x, d, h, w, p]
    """
    res = []
    config = self.cfg
    anchor_ids = self.anchor_ids
    anchor_params = self.anchor_params
    anchor_centers = []

    for a in anchor_ids:
      # category starts from 1 with 0 denoting background
      # id starts from 0
      cat = a + 1
      dets = detections[detections[:, -1] == cat]
      if len(dets):
        b, p, z, y, x, d, h, w, _ = dets[0]
        anchor_centers.append([z, y, x])
        res.append(dets[0])
      else:
        # Does not have anchor box
        return detections

    pred_cats = np.unique(detections[:, -1]).astype(np.uint8)
    for cat in pred_cats:
      if cat - 1 not in anchor_ids:
        cat = int(cat)
        preds = detections[detections[:, -1] == cat]
        score = np.zeros((len(preds),))
        roi_name = config['roi_names'][cls_ind][cat - 1]

        for k, params in enumerate(anchor_params):
          param = params[roi_name]
          for i, det in enumerate(preds):
            b, p, z, y, x, d, h, w, _ = det
            d = np.array([z, y, x]) - np.array(anchor_centers[k])
            prob = norm.pdf(d, param[0], param[1])
            prob = np.log(prob)
            prob = np.sum(prob)
            score[i] += prob

        res.append(preds[score == score.max()][0])

    res = np.array(res)
    return res


if __name__ == '__main__':
  net = SGDANoduleNet_wt_cross_attention_wt_v2droppath_bottom(config)

  input = torch.rand([4, 1, 128, 128, 128])
  input = Variable(input)
