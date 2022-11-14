from net.uni_layer import *

from configs.sgda_config import config
import copy
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm
from net.sgda.domain_attention_module import DomainAttention
from net.sgda_sanet import da_resnet_middle_top

bn_momentum = 0.1
affine = True

class DAResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride=1, fixed_block=False, reduction=16):
        super(DAResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.domain_attention = DomainAttention(n_out, reduction=16, nclass_list=config['num_class'], fixed_block=fixed_block,
                                                groups=config['groups'], mode='noG_yesD', attention_dir=True)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
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
        out = self.domain_attention(out)

        out += residual
        out = self.relu(out)
        return out


class DAFeatureNet(nn.Module):
    def __init__(self, config):
        super(DAFeatureNet, self).__init__()
        self.resnet50 = da_resnet_middle_top.da_resnet50()

        self.back1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.reduce1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        self.reduce4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1, out1, out2, out3, out4 = self.resnet50(x)

        out4 = self.reduce1(out4)
        rev3 = self.path1(out4)
        out3 = self.reduce2(out3)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)
        out2 = self.reduce3(out2)
        comb2 = self.back2(torch.cat((rev2, out2), 1))

        return [x1, out1, comb2], out1

class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.ModuleList([nn.Conv3d(in_channels, 64, kernel_size=1)  for i in range(config['num_datasets'])])
        self.relu = nn.ReLU()
        self.logits = nn.ModuleList([nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1) for i in range(config['num_datasets'])])
        self.deltas = nn.ModuleList([nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1) for i in range(config['num_datasets'])])

    def forward(self, f, cls_ind):
        # out = self.drop(f)
        if config['rpn_univ']:
            cls_ind = 0
        else:
            cls_ind = cls_ind

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
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.ModuleList([nn.Linear(in_channels * self.crop_size[i][0] * self.crop_size[i][1] * self.crop_size[i][2], 512) \
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

class MaskHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(MaskHead, self).__init__()
        self.num_class = cfg['num_class']

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back2 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        for i in range(self.num_class):
            setattr(self, 'logits' + str(i + 1), nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, detections, features):
        img, f_2, f_4 = features  

        # Squeeze the first dimension to recover from protection on avoiding split by dataparallel      
        img = img.squeeze(0)
        f_2 = f_2.squeeze(0)
        f_4 = f_4.squeeze(0)

        _, _, D, H, W = img.shape
        out = []

        for detection in detections:
            b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection

            up1 = f_4[b, :, z_start / 4:z_end / 4, y_start / 4:y_end / 4, x_start / 4:x_end / 4].unsqueeze(0)
            up2 = self.up2(up1)
            up2 = self.back2(torch.cat((up2, f_2[b, :, z_start / 2:z_end / 2, y_start / 2:y_end / 2, x_start / 2:x_end / 2].unsqueeze(0)), 1))
            up3 = self.up3(up2)
            im = img[b, :, z_start:z_end, y_start:y_end, x_start:x_end].unsqueeze(0)
            up3 = self.back3(torch.cat((up3, im), 1))

            logits = getattr(self, 'logits' + str(int(cat)))(up3)
            logits = logits.squeeze()
 
            mask = Variable(torch.zeros((D, H, W))).cuda()
            mask[z_start:z_end, y_start:y_end, x_start:x_end] = logits
            mask = mask.unsqueeze(0)
            out.append(mask)

        out = torch.cat(out, 0)

        return out


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
    def __init__(self, cfg, rcnn_crop_size, in_channels = 128):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size']

        self.up2_1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up2_2 = nn.ModuleList([nn.Conv3d(in_channels, 64, kernel_size=1, padding=0) for i in range(config['num_datasets'])])
        self.up2_3 = nn.ModuleList([nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine) for i in range(config['num_datasets'])])
        self.up2_4 = nn.ReLU(inplace=True)
        self.up3_1 = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True)
        self.up3_2 = nn.ModuleList([nn.Conv3d(64, 64, kernel_size=1, padding=0) for i in range(config['num_datasets'])])
        self.up3_3 = nn.ModuleList([nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine) for i in range(config['num_datasets'])])
        self.up3_4 = nn.ReLU(inplace=True)
        self.back2_1 = nn.ModuleList([nn.Conv3d(128, 64, kernel_size=1, padding=0) for i in range(config['num_datasets'])])
        self.back2_2 = nn.ModuleList([nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine) for i in range(config['num_datasets'])])
        self.back2_3 = nn.ReLU(inplace=True)
        self.back3_1 = nn.ModuleList([nn.Conv3d(65, 64, kernel_size=3, padding=1) for i in range(config['num_datasets'])])
        self.back3_2 = nn.ModuleList([nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine) for i in range(config['num_datasets'])])
        self.back3_3 = nn.ReLU(inplace=True)

    def forward(self, f, inputs, proposals, cls_ind):
        img, out1, comb2 = f
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        img = img.squeeze(0)
        out1 = out1.squeeze(0)
        comb2 = comb2.squeeze(0)

        crops = []
        for p in proposals:
            b, z_start, y_start, x_start, z_end, y_end, x_end = p

            # Slice 0 dim, should never happen
            c0 = np.array(torch.Tensor([z_start, y_start, x_start]))
            c1 = np.array(torch.Tensor([z_end, y_end, x_end]))
            if np.any((c1 - c0) < 1): #np.any((c1 - c0).cpu().data.numpy() < 1):
                # c0=c0+1
                # c1=c1+1
                for i in range(3):
                    if c1[i] == 0:
                        c1[i] = c1[i] + 4
                    if c1[i] - c0[i] == 0:
                        c1[i] = c1[i] + 4
                print(p)
                print('c0:', c0, ', c1:', c1)
            z_end, y_end, x_end = c1

            fe1 = comb2[int(b), :, int(z_start / 4):int(z_end / 4), int(y_start / 4):int(y_end / 4), int(x_start / 4):int(x_end / 4)].unsqueeze(0)
            fe1_up = self.up2_1(fe1)
            fe1_up = self.up2_2[cls_ind](fe1_up)
            fe1_up = self.up2_3[cls_ind](fe1_up)
            fe1_up = self.up2_4(fe1_up)

            fe2 = self.back2_1[cls_ind](torch.cat((fe1_up, out1[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)), 1))
            fe2 = self.back2_2[cls_ind](fe2)
            fe2 = self.back2_3(fe2)
            # fe2_up = self.up3(fe2)

            # im = img[b, :, z_start / 2:z_end / 2, y_start / 2:y_end / 2, x_start / 2:x_end / 2].unsqueeze(0)
            # up3 = self.back3(torch.cat((fe2, im), 1))
            # crop = up3.squeeze()

            crop = fe2.squeeze()
            # crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size[cls_ind])
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class SGDASANetv2_middle_top(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(SGDASANetv2_middle_top, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.feature_net = DAFeatureNet(config)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        # self.rpn_loss = Loss(cfg['num_hard'])
        

    def forward(self, inputs, cls_ind, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
        features, feat_4 = data_parallel(self.feature_net, (inputs)); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, (fs, cls_ind))

        b,D,H,W,_,num_class = self.rpn_logits_flat.shape

        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1);#print('rpn_logit ', self.rpn_logits_flat.shape)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6);#print('rpn_delta ', self.rpn_deltas_flat.shape)


        self.rpn_window    = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []
        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat, cls_ind)
            # print 'length of rpn proposals', self.rpn_proposals.shape

        if self.mode in ['train', 'valid']:
            # self.rpn_proposals = torch.zeros((0, 8)).cuda()
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels, cls_ind)

            if self.use_rcnn:
                # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels, cls_ind=cls_ind)

        #rcnn proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:
                proposal = self.rpn_proposals[:, [0, 2, 3, 4, 5, 6, 7]].cpu().numpy().copy()
                proposal[:, 1:] = center_box_to_coord_box(proposal[:, 1:])
                proposal = proposal.astype(np.int64)
                proposal[:, 1:] = ext2factor(proposal[:, 1:], 4)
                proposal[:, 1:] = clip_boxes(proposal[:, 1:], inputs.shape[2:])
                # rcnn_crops = self.rcnn_crop(features, inputs, torch.from_numpy(proposal).cuda())
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                rcnn_crops = data_parallel(self.rcnn_crop, (features, inputs, torch.from_numpy(proposal).cuda(), cls_ind))
                # rcnn_crops = self.rcnn_crop(fs, inputs, self.rpn_proposals)
                # self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
                self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, (rcnn_crops, cls_ind))
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals,
                                                                        self.rcnn_logits, self.rcnn_deltas, cls_ind=cls_ind)

                if self.mode in ['eval']:
                    #     Ensemble
                    fpr_res = get_probability(self.cfg, self.mode, inputs, self.rpn_proposals, self.rcnn_logits,
                                              self.rcnn_deltas, cls_ind=cls_ind)
                    if self.ensemble_proposals.shape[0] == fpr_res.shape[0]:
                        self.ensemble_proposals[:, 1] = (self.ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2
                    return self.rpn_proposals, self.detections, self.ensemble_proposals
            return self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, \
                   self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets
        return self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights


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

    def crf(self, detections):
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
                roi_name = config['roi_names'][cat - 1]

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
    net = SGDASANetv2_middle_top(config)

    input = torch.rand([2, 1, 304, 384, 384])
    input = Variable(input)
    net(input, None, None)

