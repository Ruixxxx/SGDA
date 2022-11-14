from dataset.datasets_info import get_datasets_info
from dataset.datasets_info import  univ_info
from configs.config_wt_adapter import config, train_config, net_config
from net.sgda_nodulenet.NoduleNet_wt_adapter import NoduleNet_wt_adapter
from net.sgda_nodulenet.NoduleNet_wt_SNR import NoduleNet_wt_SNR
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.uni_bbox_reader import BboxReader
from utils.util import Logger
import pprint
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import traceback
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import gc
import logging
import pandas as pd
from evaluationScript.uni_noduleCADEvaluation import noduleCADEvaluation

this_module = sys.modules[__name__]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Nodule Detection network')
    parser.add_argument('--dataset', dest='dataset', help='training dataset',
                        default=train_config['dataset'], type=str)
    parser.add_argument('--net', '-m', metavar='NET', help='neural net',
                        default=train_config['net'])
    parser.add_argument("--mode", type=str, default='eval',
                        help="you want to test or val")
    parser.add_argument('--datasets_list', nargs='+', help='datasets list for training',
                        default=train_config['datasets_list'], type=str)
    # parser.add_argument('--start_epoch', dest='start_epoch',help='starting epoch',
    #                     default=train_config['start_epoch'], type=int)
    # parser.add_argument('--epochs', dest='epochs',help='number of epochs to train',
    #                     default=train_config['epochs'], type=int)
    # # parser.add_argument('--disp_interval', dest='disp_interval',
    # #                     help='number of iterations to display',
    # #                     default=50, type=int)
    # parser.add_argument('--epoch_save', dest='epoch_save',help='save frequency',
    #                     default=train_config['epoch_save'], type=int)
    # parser.add_argument('--epoch_rcnn', dest='epoch_rcnn', help='number of epochs before training rcnn',
    #                     default=train_config['epoch_rcnn'], type=int)
    # # parser.add_argument('--bs', dest='batch_size', help='batch size',
    # #                     default=train_config['batch_size'], type=int)
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data',
                        default=train_config['num_workers'], type=int)
    # parser.add_argument('--backward_together', dest='backward_together',
    #                     help='whether use original backward method, will update optimizer \
    #                          when every dataset is finished if choose False',
    #                     default=train_config['backward_together'], type=int)
    # parser.add_argument('--randomly_chosen_datasets', dest='randomly_chosen_datasets',
    #                     help='Whether randomly choose datasets',
    #                     default=train_config['randomly_chosen_datasets'], type=int)
    # parser.add_argument('--class_agnostic', dest='class_agnostic',
    #                     help='whether perform class_agnostic bbox regression',
    #                     default=train_config['class_agnostic'], type=int)
    #
    # # config optimization
    # parser.add_argument('--init_lr', dest='init_lr', help='initial learning rate',
    #                     default=train_config['init_lr'], type=float)
    # parser.add_argument('--o', dest='optimizer', help='training optimizer',
    #                     default=train_config['optimizer'], type=str)
    # parser.add_argument('--momentum', dest='momentum', help='momentum',
    #                     default=train_config['momentum'], type=float)
    # parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay (default: 1e-4)',
    #                     default=train_config['weight_decay'], type=float)
    #
    # # save trained model
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models',
                        default=train_config['out_dir'], nargs=argparse.REMAINDER)
    #
    # # resume trained model
    # # parser.add_argument('--r', dest='resume', help='resume checkpoint or not',
    # #                     default=False, type=bool)
    parser.add_argument('--ckpt', dest='ckpt', help='checkpoint to use',
                        default=train_config['load_checkpoint'], type=str)

    # GPU configuration
    # parser.add_argument('--cuda', dest='cuda',
    #                     help='whether use CUDA',
    #                     action='store_true')
    # # parser.add_argument('--ls', dest='large_scale',
    # #                     help='whether use large image scale',
    # #                     action='store_true')
    # parser.add_argument('--mGPUs', dest='mGPUs',
    #                     help='whether use multiple GPUs',
    #                     action='store_true')
    #
    # # other
    # parser.add_argument('--use_tfboard', dest='use_tfboard', help='whether use tensorflow tensorboard',
    #                     default=True, type=bool)
    # parser.add_argument('--random_resize', dest='random_resize', help='whether randomly resize images',
    #                     default="True", type=str)
    # parser.add_argument('--fix_bn', dest='fix_bn', help='whether fix batch norm layer',
    #                     default="True", type=str)
    parser.add_argument('--use_mux', dest='use_mux', help='whether use BN MUX',
                        default="False", type=str)
    # parser.add_argument('--update_chosen', dest='update_chosen',
    #                     help='Whether update chosen layers',
    #                     default="False", type=str)
    # parser.add_argument('--warmup_steps', dest='warmup_steps',
    #                         help='Whether use warm up',
    #                         default=0, type=int)
    # parser.add_argument('--less_blocks', dest='less_blocks',
    #                     help='Whether use less blocks',
    #                     default='False', type=str)
    parser.add_argument('--num_adapters', dest='num_adapters',
                        help='Number of se layers adapter',
                        default=5, type=int)
    parser.add_argument('--rpn_univ', dest='rpn_univ',
                        help='Whether use universal rpn',
                        default=train_config['rpn_univ'], type=str)
    args = parser.parse_args()
    return args


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# Check whether although gradients is zero, data is still updated
def check_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'RCNN_cls_score_layers' in name and 'weight' in name:
                print(name, torch.sum(param.grad.data)*1000, torch.sum(param)*1000)

# Get datasets from datasets_list
def get_dataset_list(datasets_list, train_lists, label_types_list, data_dirs, config):
    train_dataset_list = []
    for i in range(len(datasets_list)):
        train_dataset = []
        for j in range(len(train_lists[i])):
            train_list = train_lists[i][j]
            label_type = label_types_list[i][j]
            data_dir = data_dirs[i]
            if label_type == 'bbox':
                dataset = BboxReader(data_dir, train_list, config, mode='train')
            train_dataset.append(dataset)
        train_dataset_list.append(train_dataset)
    return train_dataset_list

# Get datasets from datasets_list
def get_dataset_list1(datasets_list, test_names, label_types_list, augtypes, data_dirs, config):
    test_dataset_list = []
    for i in range(len(datasets_list)):
        test_name = test_names[i]
        label_type = label_types_list[i][0]
        data_dir = data_dirs[i]
        augtype = augtypes[i]
        if label_type == 'bbox':
            dataset = BboxReader(data_dir, test_name, augtype, config, mode='eval')
        test_dataset_list.append(dataset)
    return test_dataset_list

# Get dataloaders from datasets
def get_dataloader_list(datasets_list, train_dataset_list, batch_sizes, num_workers):
    train_loader_list = []
    train_iters_per_epoch_list = []
    for i in range(len(datasets_list)):
        train_loader = DataLoader(ConcatDataset(train_dataset_list[i]), batch_size=batch_sizes[i], shuffle=True,
                                  num_workers=num_workers, pin_memory=True, collate_fn=train_collate)
        train_loader_list.append(train_loader)

        train_iters_per_epoch = int(len(train_dataset_list[i][0]) / batch_sizes[i]) + 1
        train_iters_per_epoch_list.append(train_iters_per_epoch)
    return train_loader_list, train_iters_per_epoch_list

def get_dataloader_list1(datasets_list, test_dataset_list, num_workers):
    test_loader_list = []
    test_iters_per_epoch_list = []
    for i in range(len(datasets_list)):
        test_loader = DataLoader(test_dataset_list[i], batch_size=1, shuffle=False,
                                  num_workers=num_workers, pin_memory=False, collate_fn=train_collate)
        test_loader_list.append(test_loader)

        test_iters_per_epoch = int(len(test_dataset_list[i]))
        test_iters_per_epoch_list.append(test_iters_per_epoch)
    return test_loader_list, test_iters_per_epoch_list

# Get randomly shuffled datasets ids and total iters
def randomly_chosen_dataset(train_iters_per_epoch_list):
    train_total_iters = 0
    for iters_num in train_iters_per_epoch_list:
        train_total_iters += iters_num
    train_datasets_ids = list(range(train_total_iters))
    datasets_start = 0
    for index, iters in enumerate(train_iters_per_epoch_list):
        train_datasets_ids[datasets_start:(datasets_start + iters)] = [index for temp in list(range(iters))]
        datasets_start += iters
    np.random.shuffle(train_datasets_ids)
    return train_datasets_ids, train_total_iters

def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    # Load training configuration
    args = parse_args()

    if args.mode == 'eval':
        dataset = args.dataset
        net = args.net

        # start_epoch = args.start_epoch
        # epochs = args.epochs
        # epoch_save = args.epoch_save
        # epoch_rcnn = args.epoch_rcnn
        # batch_size = args.batch_size
        num_workers = args.num_workers

        # init_lr = args.init_lr
        # lr_schedule = train_config['lr_schedule']
        # optimizer = args.optimizer
        # momentum = args.momentum
        # weight_decay = args.weight_decay

        out_dir = args.save_dir
        initial_checkpoint = args.ckpt

        print('Called with args:')
        print(args)

        datasets_list = args.datasets_list
        train_lists = config['train_lists']
        val_lists =  config['val_lists']
        test_names = config['test_names']
        data_dirs = config['data_dirs']
        annotation_dirs = config['annotation_dirs']
        label_types_list = config['label_types_list']
        augtypes = config['augtype']
        roi_names = config['roi_names']
        num_class = config['num_class']
        crop_size = config['crop_size']
        bbox_border = config['bbox_border']
        pad_value = config['pad_value']

        # torch.backends.cudnn.benchmark = True
        # if torch.cuda.is_available() and not args.cuda:
        #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        # Load data
        test_dataset_list = get_dataset_list1(datasets_list, test_names, label_types_list, augtypes, data_dirs, config)

        test_loader_list, test_iters_per_epoch_list = get_dataloader_list1(datasets_list, test_dataset_list, num_workers)

        for i in range(len(datasets_list)):
            print('test_iters_per_epoch for datasets {%s} is: {%d}'%(datasets_list[i], len(test_dataset_list[i])))

        # Initialize network
        net = getattr(this_module, net)(config)
        net = net.cuda()

        # print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        # Training
        print('out_dir', out_dir)
        save_dir_list = []
        for i in datasets_list:
            save_dir = os.path.join(out_dir, '{}_res'.format(i), str(epoch))
            save_dir_list.append(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(os.path.join(save_dir, 'FROC')):
                os.makedirs(os.path.join(save_dir, 'FROC'))

        for i in range(len(datasets_list)):
            eval(net, i, test_loader_list[i], annotation_dirs[i], data_dirs[i], save_dir_list[i])

    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, cls_ind, dataset, annotation_dir, data_dir, save_dir=None):
    net.set_mode('eval')
    net.use_rcnn = True

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels) in tqdm(enumerate(dataset), total=len(dataset), desc='eval'):
        try:
            input = Variable(input).cuda()
            truth_bboxes = np.array(truth_bboxes)
            truth_labels = np.array(truth_labels)
            pid = dataset.dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid))

            with torch.no_grad():
                # input = input.cuda().unsqueeze(0)
                rpn_proposals, detections, ensemble_proposals = net.forward(input, cls_ind, truth_bboxes, truth_labels)

            rpns = rpn_proposals.cpu().numpy()
            detections = detections.cpu().numpy()
            ensembles = ensemble_proposals.cpu().numpy()

            print('rpn', rpns.shape)
            print('detection', detections.shape)
            print('ensemble', ensembles.shape)

            if len(rpns):
                rpns = rpns[:, 1:]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % (pid)), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % (pid)), ensembles)

            # Clear gpu memory
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()
            traceback.print_exc()

            print
            return

    # Generate prediction csv for the use of performing FROC analysis
    # Save both rpn and rcnn results
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            # rpns[:, 4] = (rpns[:, 4] + rpns[:, 5] + rpns[:, 6]) / 3
            # rpns = rpns[:, [3, 2, 1, 4, 0]]
            rpns = rpns[:, [3, 2, 1, 6, 5, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            # rcnns[:, 4] = (rcnns[:, 4] + rcnns[:, 5] + rcnns[:, 6]) / 3
            # rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            rcnns = rcnns[:, [3, 2, 1, 6, 5, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            # ensembles[:, 4] = (ensembles[:, 4] + ensembles[:, 5] + ensembles[:, 6]) / 3
            # ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            ensembles = ensembles[:, [3, 2, 1, 6, 5, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))

    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['series_id', 'x_center', 'y_center', 'z_center', 'w_mm', 'h_mm', 'd_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')

    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))

    # noduleCADEvaluation(annotation_dir, data_dir, dataset.dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))
    #
    # noduleCADEvaluation(annotation_dir, data_dir, dataset.dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    noduleCADEvaluation(annotation_dir, data_dir, dataset.dataset.set_name, ensemble_submission_path,
                            os.path.join(eval_dir, 'ensemble'))

    print

def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]

    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks

if __name__ == '__main__':
    main()
