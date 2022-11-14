import os
import numpy as np
import torch
import random
from dataset.datasets_info import univ_info


# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# filename = 'SE_NoduleNet'
# filename = 'SE_SANet'
# filename = 'SGSE_NoduleNet'
filename = 'SGSE_SANet'

def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors


bases = [5, 10, 20, 30, 50]
aspect_ratios = [[1, 1, 1]]


def lr_shedule(epoch, init_lr=0.01, total=200):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr


train_config = {
    #### Datasets
    # 'dataset': 'luna16_russia',
    # 'datasets_list': ['luna16_sgda_aug', 'russia_sgda_aug'],
    # 'dataset': 'luna16_tianchi',
    # 'datasets_list': ['luna16_sgda_aug', 'tianchi_sgda_aug'],
    # 'dataset': 'russia_tianchi',
    # 'datasets_list': ['russia_sgda_aug', 'tianchi_sgda_aug'],

    # 'dataset': 'luna16_finetune0',
    # 'datasets_list': ['luna16_finetune0'],
    # 'dataset': 'russia_finetune0',
    # 'datasets_list': ['russia_finetune0'],
    # 'dataset': 'tianchi_finetune0',
    # 'datasets_list': ['tianchi_finetune0'],
    # 'dataset': 'luna16_finetune1',
    # 'datasets_list': ['luna16_finetune1'],
    # 'dataset': 'russia_finetune1',
    # 'datasets_list': ['russia_finetune1'],
    # 'dataset': 'tianchi_finetune1',
    # 'datasets_list': ['tianchi_finetune1'],
    # 'dataset': 'luna16_finetune2',
    # 'datasets_list': ['luna16_finetune2'],
    # 'dataset': 'russia_finetune2',
    # 'datasets_list': ['russia_finetune2'],
    # 'dataset': 'tianchi_finetune2',
    # 'datasets_list': ['tianchi_finetune2'],
    # 'dataset': 'luna16_finetune3',
    # 'datasets_list': ['luna16_finetune3'],
    # 'dataset': 'russia_finetune3',
    # 'datasets_list': ['russia_finetune3'],
    # 'dataset': 'tianchi_finetune3',
    # 'datasets_list': ['tianchi_finetune3'],
    # 'dataset': 'luna16_sgda_aug',
    # 'datasets_list': ['luna16_sgda_aug'],
    # 'dataset': 'russia_sgda_aug',
    # 'datasets_list': ['russia_sgda_aug'],
    # 'dataset': 'tianchi_sgda_aug',
    # 'datasets_list': ['tianchi_sgda_aug'],

    # 'dataset': 'universal_sgda_aug',
    # 'datasets_list': ['luna16_sgda_aug', 'russia_sgda_aug', 'tianchi_sgda_aug'],
    'dataset': 'pn9_sanet',
    'datasets_list': ['pn9_sanet'],

    #### NoduleNet
    # 'net': 'uniNoduleNet',
    # 'net_name': 'uniNoduleNet',
    # 'net': 'SEuniNoduleNet',                                                      #87
    # 'net_name': 'SEuniNoduleNet',
    # 'net': 'SENoduleNet',                                                         #87
    # 'net_name': 'SENoduleNet',
    # 'net': 'DANoduleNet',                                                         #87
    # 'net_name': 'DANoduleNet',

    # 'net': 'SGuniNoduleNet',
    # 'net_name': 'SGuniNoduleNet',
    # 'net': 'SGSENoduleNet',
    # 'net_name': 'SGSENoduleNet',
    # 'net': 'SGDANoduleNetv2',
    # 'net_name': 'SGDANoduleNetv2',
    # 'net': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath',
    # 'net_name': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath',

    #### ablation study for locations
    # 'net': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath_bottom',
    # 'net_name': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath_bottom',
    # 'net': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath_middle',
    # 'net_name': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath_middle',
    # 'net': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath_top',
    # 'net_name': 'SGDANoduleNet_wt_cross_attention_wt_v2droppath_top',

    #### ablation study for directions
    # 'net': 'SGDANoduleNetv4',
    # 'net_name': 'SGDANoduleNetv4',                                                  #115
    # 'd': 0, #1 #2

    #### SANet
    # 'net': 'DASANet,                                                                #115
    # 'net_name': 'DASANet',
    # 'net': 'SGDASANetv2_middle_top',                                                #87
    # 'net_name': 'SGDASANetv2_middle_top',
    'net': 'SGDASANet_wt_v2droppath_middle_top',                                    #115
    'net_name': 'SGDASANet_wt_v2droppath_middle_top',

    'groups': 1, #4
    'use_mux': False,
    'num_adapters': 1, #3
    'drop_path': 0.2,
    'rpn_univ': False,

    'start_epoch': 1,
    'epochs': 400, # 400
    'epoch_save': 1,
    'epoch_rcnn': 10, #10 #80 #90
    # 'batch_size': 16,
    'num_workers': 8,
    'backward_together': 0,
    'randomly_chosen_datasets': 1,
    'class_agnostic': 1,

    'lr_schedule': lr_shedule,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4,
}

net_config = {
    # Net configuration
    'num_datasets': len(train_config['datasets_list']),
    'anchors': get_anchors(bases, aspect_ratios),
    'chanel': 1,
    # 'crop_size': data_config['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    'blacklist': [],

    # 'augtype': {'flip': False, 'rotate': False, 'scale': True, 'swap': False},
    'augtype': univ_info(train_config['datasets_list'], 'augtype'),
    'r_rand_crop': 0.,
    'pad_value': 170,

    # region proposal network configuration
    'rpn_train_bg_thresh_high': univ_info(train_config['datasets_list'], 'rpn_train_bg_thresh_high'), #0.02
    'rpn_train_fg_thresh_low': univ_info(train_config['datasets_list'], 'rpn_train_fg_thresh_low'), #0.5

    'rpn_train_nms_num': univ_info(train_config['datasets_list'], 'rpn_train_nms_num'), #300
    'rpn_train_nms_pre_score_threshold': univ_info(train_config['datasets_list'], 'rpn_train_nms_pre_score_threshold'), #0.5
    'rpn_train_nms_overlap_threshold': univ_info(train_config['datasets_list'], 'rpn_train_nms_overlap_threshold'), #0.1
    'rpn_test_nms_pre_score_threshold': univ_info(train_config['datasets_list'], 'rpn_test_nms_pre_score_threshold'), #0.52, #0.5
    'rpn_test_nms_overlap_threshold': univ_info(train_config['datasets_list'], 'rpn_test_nms_overlap_threshold'), #0.01, #0.1

    # false positive reduction network configuration
    # 'num_class': len(data_config['roi_names']) + 1, #需要修改
    'rcnn_crop_size': univ_info(train_config['datasets_list'], 'rcnn_crop_size'), #(7, 7, 7) can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': univ_info(train_config['datasets_list'], 'rcnn_train_fg_thresh_low'), #0.5
    'rcnn_train_bg_thresh_high': univ_info(train_config['datasets_list'], 'rcnn_train_bg_thresh_high'), #0.1
    'rcnn_train_batch_size': univ_info(train_config['datasets_list'], 'rcnn_train_batch_size'), #64
    'rcnn_train_fg_fraction': univ_info(train_config['datasets_list'], 'rcnn_train_fg_fraction'), #0.5
    'rcnn_train_nms_pre_score_threshold': univ_info(train_config['datasets_list'], 'rcnn_train_nms_pre_score_threshold'), #0.5
    'rcnn_train_nms_overlap_threshold': univ_info(train_config['datasets_list'], 'rcnn_train_nms_overlap_threshold'), #0.1
    'rcnn_test_nms_pre_score_threshold': univ_info(train_config['datasets_list'], 'rcnn_test_nms_pre_score_threshold'), #0.5, #0.0
    'rcnn_test_nms_overlap_threshold': univ_info(train_config['datasets_list'], 'rcnn_test_nms_overlap_threshold'), #0.01

    'box_reg_weight': [1., 1., 1., 1., 1., 1.]

}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3

# if train_config['net_name'] == 'DANoduleNet':
#     train_config['net_name'] = '{}_{}'.format(train_config['net_name'], train_config['num_adapters'])
if train_config['net_name'] == 'SGDANoduleNet_wt_cross_attention_wt_v2droppath':
    train_config['net_name'] = train_config['net_name'].replace('path', str(train_config['drop_path']))
if train_config['num_adapters'] != 1:
    train_config['net_name'] = '{}_a{}'.format(train_config['net_name'], train_config['num_adapters'])
if train_config['groups'] != 1:
    train_config['RESULTS_DIR'] = os.path.join('/data/uod_model/uod_model_115/{}/{}_g{}_results'.format(
        filename, train_config['net_name'], train_config['groups']), train_config['dataset'])
else:
    train_config['RESULTS_DIR'] = os.path.join('/data/uod_model/uod_model_115/{}/{}_results'.format(
        filename, train_config['net_name']), train_config['dataset'])
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'cross_val_test_{}'.format(train_config['epoch_rcnn']))
train_config['load_checkpoint'] = os.path.join(train_config['out_dir'], 'model/350.ckpt')
train_config['initial_checkpoint'] = None
data_config = {
    'train_lists': univ_info(train_config['datasets_list'], 'train_list'),
    'val_lists': univ_info(train_config['datasets_list'], 'val_list'),
    'test_names': univ_info(train_config['datasets_list'], 'test_name'),
    'data_dirs': univ_info(train_config['datasets_list'], 'data_dir'),
    'annotation_dirs': univ_info(train_config['datasets_list'], 'annotation_dir'),
    'batch_sizes': univ_info(train_config['datasets_list'], 'BATCH_SIZE'),
    'label_types_list': univ_info(train_config['datasets_list'], 'label_types'),
    'roi_names': univ_info(train_config['datasets_list'], 'roi_names'),
    'num_class': [len(i) + 1 for i in univ_info(train_config['datasets_list'], 'roi_names')],
    'crop_size': univ_info(train_config['datasets_list'], 'crop_size')[0],
    'bbox_border': univ_info(train_config['datasets_list'], 'bbox_border')[0],
    'pad_value': univ_info(train_config['datasets_list'], 'pad_value')[0]
}

config = dict(data_config, **net_config)
config = dict(config, **train_config)
