
def get_datasets_info(dataset, use_dict = False, test=False):
    datasets_info = dict()
    datasets_info['dataset'], datasets_info['train_list'], datasets_info['val_list'], datasets_info['test_name'], \
    datasets_info['data_dir'], datasets_info['annotation_dir'], datasets_info['BATCH_SIZE'], datasets_info['label_types'], \
    datasets_info['roi_names'], datasets_info['crop_size'], datasets_info['bbox_border'], datasets_info['pad_value'], \
    datasets_info['augtype'], \
    datasets_info['rpn_train_bg_thresh_high'], datasets_info['rpn_train_fg_thresh_low'], \
    datasets_info['rpn_train_nms_num'], datasets_info['rpn_train_nms_pre_score_threshold'], \
    datasets_info['rpn_train_nms_overlap_threshold'], datasets_info['rpn_test_nms_pre_score_threshold'], \
    datasets_info['rpn_test_nms_overlap_threshold'], datasets_info['rcnn_crop_size'], \
    datasets_info['rcnn_train_fg_thresh_low'], datasets_info['rcnn_train_bg_thresh_high'], \
    datasets_info['rcnn_train_batch_size'], datasets_info['rcnn_train_fg_fraction'], \
    datasets_info['rcnn_train_nms_pre_score_threshold'], datasets_info['rcnn_train_nms_overlap_threshold'], \
    datasets_info['rcnn_test_nms_pre_score_threshold'], datasets_info['rcnn_test_nms_overlap_threshold'] \
           = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, \
             None, None, None, None, None, None, None, None, None, None, None, None
    if dataset == 'luna16_sgda_aug':
        datasets_info['dataset'] = 'luna16_sgda_aug'
        datasets_info['train_list'] = ['/home/xurui/data0/luna16/sgda_split/train.csv']
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.1 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.05 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.05 #0.1
    elif dataset == 'russia_sgda_aug':
        datasets_info['dataset'] = 'russia_sgda_aug'
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.7 #0.8 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.04 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.04 #0.1
    elif dataset == 'tianchi_sgda_aug':
        datasets_info['dataset'] = 'tianchi_sgda_aug'
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.1 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.05 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.05 #0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0 #0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0 #0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0 #0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5 #0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1 #0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0 #0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1
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
        #
        datasets_info['rpn_train_bg_thresh_high'] = 0.02
        datasets_info['rpn_train_fg_thresh_low'] = 0.5
        datasets_info['rpn_train_nms_num'] = 300
        datasets_info['rpn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rpn_test_nms_pre_score_threshold'] = 0.5
        datasets_info['rpn_test_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_crop_size'] = (7, 7, 7)
        datasets_info['rcnn_train_fg_thresh_low'] = 0.5
        datasets_info['rcnn_train_bg_thresh_high'] = 0.1
        datasets_info['rcnn_train_batch_size'] = 64
        datasets_info['rcnn_train_fg_fraction'] = 0.5
        datasets_info['rcnn_train_nms_pre_score_threshold'] = 0.5
        datasets_info['rcnn_train_nms_overlap_threshold'] = 0.1
        datasets_info['rcnn_test_nms_pre_score_threshold'] = 0.0
        datasets_info['rcnn_test_nms_overlap_threshold'] = 0.1

    if use_dict:
        return datasets_info
    else:
        return  datasets_info['dataset'], datasets_info['train_list'], datasets_info['val_list'], datasets_info['test_name'], \
                datasets_info['data_dir'], datasets_info['annotation_dir'], datasets_info['BATCH_SIZE'], datasets_info['label_types'], \
                datasets_info['roi_names'], datasets_info['crop_size'], datasets_info['bbox_border'], datasets_info['pad_value'], \
                datasets_info['augtype'], \
                datasets_info['rpn_train_bg_thresh_high'], datasets_info['rpn_train_fg_thresh_low'], \
                datasets_info['rpn_train_nms_num'], datasets_info['rpn_train_nms_pre_score_threshold'], \
                datasets_info['rpn_train_nms_overlap_threshold'], datasets_info['rpn_test_nms_pre_score_threshold'], \
                datasets_info['rpn_test_nms_overlap_threshold'], datasets_info['rcnn_crop_size'], \
                datasets_info['rcnn_train_fg_thresh_low'], datasets_info['rcnn_train_bg_thresh_high'], \
                datasets_info['rcnn_train_batch_size'], datasets_info['rcnn_train_fg_fraction'], \
                datasets_info['rcnn_train_nms_pre_score_threshold'], datasets_info['rcnn_train_nms_overlap_threshold'], \
                datasets_info['rcnn_test_nms_pre_score_threshold'], datasets_info['rcnn_test_nms_overlap_threshold']

def univ_info(datasets_list, variable):
    # print(datasets_list)
    # print(variable)
    list = [get_datasets_info(datasets, use_dict=True)[variable] for datasets in datasets_list]
    return list