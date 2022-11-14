from net.sgda_sanet.SGDASANetv2_middle_top import SGDASANetv2_middle_top
from net.sgda_sanet.SGDASANet_wt_v2droppath_middle_top import SGDASANet_wt_v2droppath_middle_top
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.uni_bbox_reader import BboxReader
from utils.util import Logger
from configs.sgda_config import train_config, data_config, net_config, config
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
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
this_module = sys.modules[__name__]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Nodule Detection network')
    # parser.add_argument('--dataset', dest='dataset', help='training dataset',
    #                     default=train_config['dataset'], type=str)
    parser.add_argument('--net', '-m', metavar='NET', help='neural net',
                        default=train_config['net'])
    parser.add_argument('--datasets_list', nargs='+', help='datasets list for training',
                        default=train_config['datasets_list'], type=str)
    # parser.add_argument('--start_epoch', dest='start_epoch',help='starting epoch',
    #                     default=train_config['start_epoch'], type=int)
    parser.add_argument('--epochs', dest='epochs',help='number of epochs to train',
                        default=train_config['epochs'], type=int)
    # parser.add_argument('--disp_interval', dest='disp_interval',
    #                     help='number of iterations to display',
    #                     default=50, type=int)
    parser.add_argument('--epoch_save', dest='epoch_save',help='save frequency',
                        default=train_config['epoch_save'], type=int)
    parser.add_argument('--epoch_rcnn', dest='epoch_rcnn', help='number of epochs before training rcnn',
                        default=train_config['epoch_rcnn'], type=int)
    parser.add_argument('--bs', dest='batch_sizes', help='batch size',
                        default=data_config['batch_sizes'], type=int)
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data',
                        default=train_config['num_workers'], type=int)
    parser.add_argument('--backward_together', dest='backward_together',
                        help='whether use original backward method, will update optimizer \
                             when every dataset is finished if choose False',
                        default=train_config['backward_together'], type=int)
    parser.add_argument('--randomly_chosen_datasets', dest='randomly_chosen_datasets',
                        help='Whether randomly choose datasets',
                        default=train_config['randomly_chosen_datasets'], type=int)
    parser.add_argument('--class_agnostic', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=train_config['class_agnostic'], type=int)

    # config optimization
    parser.add_argument('--init_lr', dest='init_lr', help='initial learning rate',
                        default=train_config['init_lr'], type=float)
    parser.add_argument('--o', dest='optimizer', help='training optimizer',
                        default=train_config['optimizer'], type=str)
    parser.add_argument('--momentum', dest='momentum', help='momentum',
                        default=train_config['momentum'], type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay (default: 1e-4)',
                        default=train_config['weight_decay'], type=float)

    # save trained model
    parser.add_argument('--out_dir', dest='out_dir', help='directory to save models',
                        default=train_config['out_dir'], nargs=argparse.REMAINDER)

    # resume trained model
    # parser.add_argument('--r', dest='resume', help='resume checkpoint or not',
    #                     default=False, type=bool)
    parser.add_argument('--ckpt', dest='ckpt', help='checkpoint to use',
                        default=train_config['initial_checkpoint'], type=str)

    # other
    parser.add_argument('--use_tfboard', dest='use_tfboard', help='whether use tensorflow tensorboard',
                        default=True, type=bool)
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
def get_dataset_list(datasets_list, train_lists, label_types_list, augtypes, data_dirs, config):
    train_dataset_list = []
    for i in range(len(datasets_list)):
        train_dataset = []
        for j in range(len(train_lists[i])):
            train_list = train_lists[i][j]
            label_type = label_types_list[i][j]
            data_dir = data_dirs[i]
            augtype = augtypes[i]
            if label_type == 'bbox':
                dataset = BboxReader(data_dir, train_list, augtype, config, mode='train')
            train_dataset.append(dataset)
        train_dataset_list.append(train_dataset)
    return train_dataset_list

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

    # Load training configuration
    args = parse_args()

    net = args.net

    # start_epoch = args.start_epoch
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    batch_sizes = args.batch_sizes
    num_workers = args.num_workers

    init_lr = args.init_lr
    lr_schedule = train_config['lr_schedule']
    optimizer = args.optimizer
    momentum = args.momentum
    weight_decay = args.weight_decay

    out_dir = args.out_dir
    initial_checkpoint = args.ckpt

    print('Called with args:')
    print(args)

    datasets_list = args.datasets_list
    train_lists = config['train_lists']
    val_lists =  config['val_lists']
    test_names = config['test_names']
    data_dirs = config['data_dirs']
    label_types_list = config['label_types_list']
    augtypes = config['augtype']

    # torch.backends.cudnn.benchmark = True
    # if torch.cuda.is_available() and not args.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Load data
    train_dataset_list = get_dataset_list(datasets_list, train_lists, label_types_list, augtypes, data_dirs, config)
    val_dataset_list = get_dataset_list(datasets_list, val_lists, label_types_list, augtypes, data_dirs, config)

    train_loader_list, train_iters_per_epoch_list = get_dataloader_list(datasets_list, train_dataset_list, batch_sizes, num_workers)
    val_loader_list, val_iters_per_epoch_list = get_dataloader_list(datasets_list, val_dataset_list, batch_sizes, num_workers)

    for i in range(len(datasets_list)):
        print('train_iters_per_epoch for datasets {%s} is: {%d}'%(datasets_list[i], train_iters_per_epoch_list[i]))
        print('val_iters_per_epoch for datasets {%s} is: {%d}' % (datasets_list[i], val_iters_per_epoch_list[i]))

    # Training options
    backward_together = args.backward_together == 1
    if backward_together:
        print('INFO: backward_together')
    else:
        print('INFO: backward after each datasets')

    randomly_chosen_datasets = args.randomly_chosen_datasets == 1
    if randomly_chosen_datasets:
        print('INFO: Randomly choose datasets during training')
    else:
        print('INFO: Train every dataset equal times')
    if randomly_chosen_datasets:
        train_datasets_ids, train_total_iters = randomly_chosen_dataset(train_iters_per_epoch_list)
        val_datasets_ids, val_total_iters = randomly_chosen_dataset(val_iters_per_epoch_list)
    else:
        train_total_iters = max(train_iters_per_epoch_list) * len(train_iters_per_epoch_list)
        val_total_iters = max(val_iters_per_epoch_list) * len(val_iters_per_epoch_list)

    # Initialize network
    net = getattr(this_module, net)(config)
    net = net.cuda()

    ad_params = list(map(id, net.feature_net.resnet50.layer2[0].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer2[2].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer2[3].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer2[5].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer3[0].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer3[2].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer3[3].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer3[5].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer3[6].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer3[8].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer4[0].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer4[1].domain_attention.parameters())) + \
                list(map(id, net.feature_net.resnet50.layer4[2].domain_attention.parameters()))

    de_params = list(map(id, net.rpn.parameters())) + \
                list(map(id, net.rcnn_head.parameters())) + \
                list(map(id, net.rcnn_crop.parameters()))

    base_params = filter(lambda p: id(p) not in ad_params + de_params, net.parameters())
    # for param in net.parameters():
    #     if id(param) not in ad_params + de_params:
    #         param.requires_grad = False

    # params = [{'params': net.feature_net.resnet50.layer1[0].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer1[1].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer1[2].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer2[0].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer2[2].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer2[3].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer2[5].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer3[0].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer3[2].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer3[3].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer3[5].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer3[6].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer3[8].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer4[0].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer4[1].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': net.feature_net.resnet50.layer4[2].domain_attention.parameters(), 'lr': 'lr'*10},
    #           {'params': base_params}]

    # print(net)
    # if args.mGPUs:
    #     net = torch.nn.DataParallel(net)
    #     net = net.cuda()
    print(sum(p.numel() for p in net.parameters()))
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(sum(q.numel() for q in filter(lambda p: id(p) in de_params, net.parameters()) if q.requires_grad))
    # print(sum(1 for p in net.state_dict() if 'rpn' in p or 'rcnn' in p))

    optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    # optimizer = optimizer(params, lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    scaler = GradScaler()

    start_epoch = 0
    state = net.state_dict()
    # for key in state.keys():
    #     print(key)
    # exit()

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        # start_epoch = checkpoint['epoch']

        # for key in checkpoint['state_dict'].keys():
        #     print(key)
        # exit()

        # if ('finetune' in initial_checkpoint):
        #     state = {}
        #
        #     # i=0
        #     for key in list(checkpoint['state_dict'].keys()):
        #         if ('rpn' in key) or ('rcnn' in key):
        #             # print(key)
        #             # i = i + 1
        #         # if key in filter(lambda p: id(p) in de_params, net.parameters()).keys():
        #             del checkpoint['state_dict'][key]
        #         else:
        #             new_key = key
        #             state[new_key] = checkpoint['state_dict'][key]
        #     # print(i)
        #     # print(state)
        #
        # else:
        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        #new_state.update(checkpoint['state_dict'])
        try:
            net.load_state_dict(state, strict=False)
            if not ('finetune' in initial_checkpoint.split('/')[-1]):
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Load something failed!')
            traceback.print_exc()
    start_epoch = start_epoch + 1

    # Training
    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train iters %d, length of validation iters %d]' % (train_total_iters, val_total_iters))

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    for i in tqdm(range(start_epoch, epochs + 1), desc='Total'):
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            lr = lr_schedule(i, init_lr=init_lr, total=epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = np.inf

        if i >= epoch_rcnn:
            net.use_rcnn = True
        else:
            net.use_rcnn = False

        print('[epoch %d, lr %f, use_rcnn: %r]' % (i, lr, net.use_rcnn))
        train(net, train_loader_list, train_iters_per_epoch_list, train_datasets_ids, train_total_iters, optimizer, \
              i, train_writer, datasets_list, randomly_chosen_datasets, backward_together, scaler)
        validate(net, val_loader_list, val_iters_per_epoch_list, val_datasets_ids, val_total_iters, \
                 i, val_writer, datasets_list, randomly_chosen_datasets)
        print

        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if i % epoch_save == 0:
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.ckpt' % i))

    writer.close()
    train_writer.close()
    val_writer.close()


def train(net, train_loader_list, train_iters_per_epoch_list, datasets_ids, train_total_iters, optimizer, \
          epoch, writer, datasets_list, randomly_chosen_datasets, backward_together, scaler):
    net.set_mode('train')
    s = time.time()
    rpn_cls_loss_list, rpn_reg_loss_list = [], []
    rcnn_cls_loss_list, rcnn_reg_loss_list = [], []
    total_loss_list = []
    rpn_stats_list = []
    rcnn_stats_list = []

    if randomly_chosen_datasets:
        np.random.shuffle(datasets_ids)

    train_loader_iter_list = []
    for train_loader in train_loader_list:
        #
        # train_loader.dataset.reset_target_size()
        train_loader_iter = iter(cycle(train_loader))
        train_loader_iter_list.append(train_loader_iter)

    loss_temp = np.zeros((len(train_loader_iter_list)))
    step_list = list(np.zeros([len(train_loader_iter_list)]))
    step_list = [0 for i in step_list]

    for step in tqdm(range(train_total_iters)):
        # build a data batch list for storing data batch from different dataloader
        if randomly_chosen_datasets:
            cls_ind = datasets_ids[step]
        else:
            cls_ind = step % len(train_iters_per_epoch_list)
        step_list[cls_ind] += 1

        try:
            data = next(train_loader_iter_list[cls_ind])
        except:
            print('INFO: All the data of datasets {%s} is trained, reset it now' % (datasets_list[cls_ind]))
            train_loader_iter_list[cls_ind] = iter(cycle(train_loader_iter_list[cls_ind]))
            data = next(train_loader_iter_list[cls_ind])

        input = Variable(data[0]).cuda()
        truth_box = np.array(data[1])
        truth_label = np.array(data[2])

        with autocast():
            if net.use_rcnn == False:
                rpn_logits_flat, rpn_deltas_flat, rpn_labels, rpn_label_weights, rpn_targets, rpn_target_weights = net(input, cls_ind, truth_box, truth_label)
            else:
                rpn_logits_flat, rpn_deltas_flat, rpn_labels, rpn_label_weights, rpn_targets, rpn_target_weights, \
                rcnn_logits, rcnn_deltas, rcnn_labels, rcnn_targets = net(input, cls_ind, truth_box, truth_label)
            if net.use_rcnn == False:
                rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, loss, rpn_stat, rcnn_stat = net.loss(rpn_logits_flat, rpn_deltas_flat, rpn_labels,
                    rpn_label_weights, rpn_targets, rpn_target_weights)
            else:
                rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, loss, rpn_stat, rcnn_stat = net.loss(
                    rpn_logits_flat, rpn_deltas_flat, rpn_labels,
                    rpn_label_weights, rpn_targets, rpn_target_weights, rcnn_logits, rcnn_deltas, rcnn_labels, rcnn_targets)

            loss_temp[cls_ind] += loss.data.item()
            print(loss.data)

        # backward after training all datasets
        if backward_together:
            if randomly_chosen_datasets:
                if step == 0:
                    optimizer.zero_grad()
                # loss.backward()
                scaler.scale(loss).backward()
                if step == len(train_total_iters) - 1:
                    # if args.update_chosen and 'data_att' in args.net:
                    #     update_chosen_se_layer(fasterRCNN, cls_ind)
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if cls_ind == 0:
                    optimizer.zero_grad()
                # loss.backward()
                scaler.scale(loss).backward()
                if cls_ind == len(train_total_iters) - 1:
                    # if args.update_chosen and 'data_att' in args.net:
                    #     update_chosen_se_layer(fasterRCNN, cls_ind)
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
        else:
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            # if args.update_chosen and 'data_att' in args.net:
            #     update_chosen_se_layer(fasterRCNN, cls_ind)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
        # exit()
        rpn_cls_loss_list.append(rpn_cls_loss.cpu().data.item())
        rpn_reg_loss_list.append(rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss_list.append(rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss_list.append(rcnn_reg_loss.cpu().data.item())
        total_loss_list.append(loss.cpu().data.item())
        rpn_stats_list.append(np.asarray(torch.Tensor(rpn_stat).cpu(), np.float32))
        if net.use_rcnn:
            rcnn_stats_list.append(rcnn_stat)
            del rcnn_stat

        del input, truth_box, truth_label
        del loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss
        del rpn_logits_flat, rpn_deltas_flat
        del rpn_stat

        if net.use_rcnn:
            del rcnn_logits, rcnn_deltas

        torch.cuda.empty_cache()

    rpn_stats_list = np.asarray(rpn_stats_list, np.float32)

    print('Train Epoch %d, total time %f, loss %f' % (epoch, time.time() - s, np.average(total_loss_list)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
          (np.average(rpn_cls_loss_list), np.average(rpn_reg_loss_list), np.average(rcnn_cls_loss_list), np.average(rcnn_reg_loss_list)))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats_list[:, 0]) / np.sum(rpn_stats_list[:, 1]),
        100.0 * np.sum(rpn_stats_list[:, 2]) / np.sum(rpn_stats_list[:, 3]),
        np.sum(rpn_stats_list[:, 1]),
        np.sum(rpn_stats_list[:, 3]),
        np.mean(rpn_stats_list[:, 4]),
        np.mean(rpn_stats_list[:, 5]),
        np.mean(rpn_stats_list[:, 6]),
        np.mean(rpn_stats_list[:, 7]),
        np.mean(rpn_stats_list[:, 8]),
        np.mean(rpn_stats_list[:, 9])))

    # Write to tensorboard
    writer.add_scalars('datasets_loss', {datasets_list[j]:(loss_temp[j]/step_list[j]) for j in range(len(datasets_list))}
                       ,epoch)
    writer.add_scalar('loss', np.average(total_loss_list), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss_list), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss_list), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss_list), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss_list), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats_list[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats_list[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats_list[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats_list[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats_list[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats_list[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats_list], np.int32)
        rcnn_stats_list = np.asarray([stat[:-1] for stat in rcnn_stats_list], np.float32)

        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats_list[:, 0]),
            np.mean(rcnn_stats_list[:, 1]),
            np.mean(rcnn_stats_list[:, 2]),
            np.mean(rcnn_stats_list[:, 3]),
            np.mean(rcnn_stats_list[:, 4]),
            np.mean(rcnn_stats_list[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats_list[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats_list[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats_list[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats_list[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats_list[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats_list[:, 5]), epoch)

    print
    print

def validate(net, val_loader_list, val_iters_per_epoch_list, datasets_ids, val_total_iters, \
          epoch, writer, datasets_list, randomly_chosen_datasets):
    net.set_mode('valid')
    s = time.time()
    rpn_cls_loss_list, rpn_reg_loss_list = [], []
    rcnn_cls_loss_list, rcnn_reg_loss_list = [], []
    total_loss_list = []
    rpn_stats_list = []
    rcnn_stats_list = []

    if randomly_chosen_datasets:
        np.random.shuffle(datasets_ids)

    val_loader_iter_list = []
    for val_loader in val_loader_list:
        #
        # val_loader.dataset.reset_target_size()
        val_loader_iter = iter(cycle(val_loader))
        val_loader_iter_list.append(val_loader_iter)

    loss_temp = np.zeros((len(val_loader_iter_list)))
    step_list = list(np.zeros([len(val_loader_iter_list)]))
    step_list = [0 for i in step_list]

    for step in tqdm(range(val_total_iters)):
        # build a data batch list for storing data batch from different dataloader
        if randomly_chosen_datasets:
            cls_ind = datasets_ids[step]
        else:
            cls_ind = step % len(val_iters_per_epoch_list)
        step_list[cls_ind] += 1

        try:
            data = next(val_loader_iter_list[cls_ind])
        except:
            print('INFO: All the data of datasets {%s} is trained, reset it now' % (datasets_list[cls_ind]))
            val_loader_iter_list[cls_ind] = iter(cycle(val_loader_iter_list[cls_ind]))
            data = next(val_loader_iter_list[cls_ind])

        input = Variable(data[0]).cuda()
        truth_box = np.array(data[1])
        truth_label = np.array(data[2])

        with torch.no_grad():
            with autocast():
                if net.use_rcnn == False:
                    rpn_logits_flat, rpn_deltas_flat, rpn_labels, rpn_label_weights, rpn_targets, rpn_target_weights = net(
                        input, cls_ind, truth_box, truth_label)
                else:
                    rpn_logits_flat, rpn_deltas_flat, rpn_labels, rpn_label_weights, rpn_targets, rpn_target_weights, \
                    rcnn_logits, rcnn_deltas, rcnn_labels, rcnn_targets = net(input, cls_ind, truth_box, truth_label)
                if net.use_rcnn == False:
                    rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, loss, rpn_stat, rcnn_stat = net.loss(
                        rpn_logits_flat, rpn_deltas_flat, rpn_labels,
                        rpn_label_weights, rpn_targets, rpn_target_weights)
                else:
                    rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, loss, rpn_stat, rcnn_stat = net.loss(
                        rpn_logits_flat, rpn_deltas_flat, rpn_labels,
                        rpn_label_weights, rpn_targets, rpn_target_weights, rcnn_logits, rcnn_deltas, rcnn_labels,
                        rcnn_targets)

                loss_temp[cls_ind] += loss.data.item()
                print(loss.data)

        rpn_cls_loss_list.append(rpn_cls_loss.cpu().data.item())
        rpn_reg_loss_list.append(rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss_list.append(rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss_list.append(rcnn_reg_loss.cpu().data.item())
        total_loss_list.append(loss.cpu().data.item())
        rpn_stats_list.append(np.asarray(torch.Tensor(rpn_stat).cpu(), np.float32))
        if net.use_rcnn:
            rcnn_stats_list.append(rcnn_stat)
            del rcnn_stat

        del input, truth_box, truth_label
        del loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss
        del rpn_logits_flat, rpn_deltas_flat
        del rpn_stat

        if net.use_rcnn:
            del rcnn_logits, rcnn_deltas

        torch.cuda.empty_cache()

    rpn_stats_list = np.asarray(rpn_stats_list, np.float32)

    print('Train Epoch %d, total time %f, loss %f' % (epoch, time.time() - s, np.average(total_loss_list)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
          (np.average(rpn_cls_loss_list), np.average(rpn_reg_loss_list), np.average(rcnn_cls_loss_list), np.average(rcnn_reg_loss_list)))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats_list[:, 0]) / np.sum(rpn_stats_list[:, 1]),
        100.0 * np.sum(rpn_stats_list[:, 2]) / np.sum(rpn_stats_list[:, 3]),
        np.sum(rpn_stats_list[:, 1]),
        np.sum(rpn_stats_list[:, 3]),
        np.mean(rpn_stats_list[:, 4]),
        np.mean(rpn_stats_list[:, 5]),
        np.mean(rpn_stats_list[:, 6]),
        np.mean(rpn_stats_list[:, 7]),
        np.mean(rpn_stats_list[:, 8]),
        np.mean(rpn_stats_list[:, 9])))

    # Write to tensorboard
    writer.add_scalars('datasets_loss', {datasets_list[j]:(loss_temp[j]/step_list[j]) for j in range(len(datasets_list))}
                       ,epoch)
    writer.add_scalar('loss', np.average(total_loss_list), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss_list), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss_list), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss_list), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss_list), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats_list[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats_list[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats_list[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats_list[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats_list[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats_list[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats_list], np.int32)
        rcnn_stats_list = np.asarray([stat[:-1] for stat in rcnn_stats_list], np.float32)

        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats_list[:, 0]),
            np.mean(rcnn_stats_list[:, 1]),
            np.mean(rcnn_stats_list[:, 2]),
            np.mean(rcnn_stats_list[:, 3]),
            np.mean(rcnn_stats_list[:, 4]),
            np.mean(rcnn_stats_list[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats_list[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats_list[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats_list[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats_list[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats_list[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats_list[:, 5]), epoch)

    print
    print


def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))


if __name__ == '__main__':
    main()