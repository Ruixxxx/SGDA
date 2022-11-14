import os
import numpy as np
import torch
import random


# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Preprocessing using preserved HU in dilated part of mask
BASE = '/data/' # make sure you have the ending '/'

# filename = 'SE_NoduleNet'
# filename = 'SE_SANet'
filename = 'SGSE_NoduleNet'
# filename = 'SGSE_SANet'

test_config = {
    'dataset_train': 'russia_sgda_aug',
    'dataset_test': 'russia_sgda_aug',
    'load_epoch': '200',

}
dataset = 'russia_sgda_aug' # dataset_test
datasets_info = {}
if dataset == 'luna16_sgda_aug':
    datasets_info['dataset'] = 'luna16_sgda_aug'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/sgda_split/train.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv' # test
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_sgda_aug':
    datasets_info['dataset'] = 'russia_diannei_aug'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/sgda_split/train.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_sgda_aug':
    datasets_info['dataset'] = 'tianchi_diannei_aug'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/sgda_split/train.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune0':
    datasets_info['dataset'] = 'luna16_finetune0'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train0.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune0':
    datasets_info['dataset'] = 'russia_finetune0'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train0.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune0':
    datasets_info['dataset'] = 'tianchi_finetune0'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train0.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune1':
    datasets_info['dataset'] = 'luna16_finetune1'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train1.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune1':
    datasets_info['dataset'] = 'russia_finetune1'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train1.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune1':
    datasets_info['dataset'] = 'tianchi_finetune1'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train1.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune2':
    datasets_info['dataset'] = 'luna16_finetune2'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train2.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune2':
    datasets_info['dataset'] = 'russia_finetune2'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train2.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune2':
    datasets_info['dataset'] = 'tianchi_finetune2'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train2.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune3':
    datasets_info['dataset'] = 'luna16_finetune3'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train3.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune3':
    datasets_info['dataset'] = 'russia_finetune3'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train3.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune3':
    datasets_info['dataset'] = 'tianchi_finetune3'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train3.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'pn9_sanet':
    datasets_info['dataset'] = 'pn9_sanet'
    datasets_info['train_list'] = ['/data0/pn9/split_full_with_nodule_9classes/train.txt']
    datasets_info['val_list'] = ['/data0/pn9/split_full_with_nodule_9classes/val.txt']
    datasets_info['test_name'] = '/data0/pn9/split_full_with_nodule_9classes/test.txt'
    datasets_info['data_dir'] = '/data0/pn9/'
    datasets_info['annotation_dir'] = '/data0/pn9/annotations.csv'
    datasets_info['BATCH_SIZE'] = 16
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': False, 'rotate': False, 'scale': True, 'swap': False}

def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors

bases = [5, 10, 20, 30, 50]
aspect_ratios = [[1, 1, 1]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'chanel': 1,
    'crop_size': datasets_info['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    # 'blacklist': ['09184', '05181','07121','00805','08832'],
    # 'blacklist': [],

    # 'augtype': {'flip': False, 'rotate': False, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,

    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,
    
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5, #0.5
    'rpn_test_nms_overlap_threshold': 0.1, #0.1

    # false positive reduction network configuration
    'num_class': len(datasets_info['roi_names']) + 1,
    'rcnn_crop_size': (7,7,7), # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 64,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1, #0.1

    'box_reg_weight': [1., 1., 1., 1., 1., 1.]
}


def lr_shedule(epoch, init_lr=0.01, total=200):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'NoduleNet',
    'net_name': 'NoduleNet',
    # 'net': 'singleSENoduleNet',
    # 'net_name': 'singleSENoduleNet',
    # 'net': 'singleSGSENoduleNet',
    # 'net_name': 'singleSGSENoduleNet',
    # 'net': 'SANet',
    # 'net_name': 'SANet',
    'batch_size': datasets_info['BATCH_SIZE'],
    'groups': 4,
    'SGSE_mode': 'noG_yesD',

    'lr_schedule': lr_shedule,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4,

    'epochs': 400, #200 #400
    'epoch_save': 1,
    'epoch_rcnn': 90, #20 #47
    'num_workers': 8, #30

}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3

if train_config['net_name'] == 'singleSGSENoduleNet':
    train_config['net_name'] = '{}_g{}'.format(train_config['net_name'], train_config['groups'])
train_config['RESULTS_DIR'] = os.path.join('/data/uod_model/uod_model_115/{}/{}_results'.format(filename, train_config['net_name']), datasets_info['dataset'])
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'cross_val_test_{}'.format(train_config['epoch_rcnn']))
# train_config['initial_checkpoint'] = None #
train_config['initial_checkpoint'] = os.path.join(train_config['out_dir'], 'model/model.ckpt')

test_config['checkpoint'] = '/data/uod_model/uod_model_115/{}/{}_results/{}/cross_val_test_{}/model/{}.ckpt'.format(
    filename, train_config['net_name'], test_config['dataset_train'], train_config['epoch_rcnn'], test_config['load_epoch'])
test_config['out_dir'] = '/data/uod_model/uod_model_115/{}/{}_results/{}/cross_val_test_{}/{}/'.format(filename,
    train_config['net_name'], test_config['dataset_train'], train_config['epoch_rcnn'], test_config['dataset_test'])

config = dict(datasets_info, **net_config)
config = dict(config, **train_config)
config = dict(config, **test_config)
