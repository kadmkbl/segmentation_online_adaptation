"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import pandas as pd

from config import cfg, assert_and_infer_cfg
from datasets import sampler
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, evaluate_eval_all, evaluate_eval_dur_train, fast_hist, save_best_acc
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random
from torchvision.transforms import Resize

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=[],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=[],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--weather', type=str, default='snow',
                    help='ACDC weather choices')
parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,0,0,0,0,0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--wt_reg_weight', type=float, default=0.0)
parser.add_argument('--relax_denom', type=float, default=2.0)
parser.add_argument('--clusters', type=int, default=50)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--cov_stat_epoch', type=int, default=5,
                    help='cov_stat_epoch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')
parser.add_argument('--use_wtloss', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--use_isw', action='store_true', default=False,
                    help='Automatic setting from wt_layer')

parser.add_argument('--dsbn', type=bool, default=False,
                    help='Whether to use DSBN in the model')
parser.add_argument('--num_domains', type=int, default=1,
                    help='The number of source domains')

parser.add_argument('--mode', type=str, default=1,
                    help='The use of this code')
parser.add_argument('--split', type=str, default=1,
                    help='The split ratio of training sets and test sets')

parser.add_argument('--task_id', type=int, default=0,
                    help='The choice of task-specific adapters when training')

parser.add_argument('--method', type=str, default="source",
                    help='The choice of Test-Adaptation')

parser.add_argument('--reset', type=bool, default=False,
                    help='Whether to reset when each env arrives')
parser.add_argument('--CoTTA_Aug', type=bool, default=False,
                    help='Whether to Aug data when implementing CoTTA')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

freeze_bn_num = 0

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

for i in range(len(args.wt_layer)):
    if args.wt_layer[i] == 1:
        args.use_wtloss = True
    if args.wt_layer[i] == 2:
        args.use_wtloss = True
        args.use_isw = True



def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    # train_loader, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)
    train_loaders, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders, train_dataset_names = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    optim, scheduler = optimizer.get_optimizer(args, net)

    # 这里主要是在处理多GPU分布式执行的东西（dataparallel with syncBN），这有一个不好的一点 —— 多卡训练的syncBN不能装载到单卡推理的BN上，所以要用如下模块才能实现推理
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # convert all attr:`BatchNorm*D` layers in the model to class:`torch.nn.SyncBatchNorm` layers
    # net = network.warp_network_in_dataparallel(net, args.local_rank) # DistributedDataParallel
    epoch = 0
    i = 0
    
    flag = 0 # for ealry_stop during training

    best_acc = 0
    ################################################################### 训练
    if args.mode == "train":
        ########## 训练模块
        print("#### iteration", i)
        torch.cuda.empty_cache()

        max_iter = args.max_iter*args.num_domains if args.dsbn else args.max_iter
        while i < max_iter:
            # Update EPOCH CTR
            cfg.immutable(False)
            cfg.ITER = i
            cfg.immutable(True)

            i = train(train_loaders, net, optim, epoch, writer, scheduler, max_iter)
            # 只有多卡训练模式下有用，以使shuffle操作能够在多个epoch中正常工作。否则，dataloader迭代器产生的数据将始终使用相同的顺序（shuffle失效）。
            # train_loader.sampler.set_epoch(epoch + 1)
            # for index in range(len(train_loaders)):
            #     train_loaders[index].sampler.set_epoch(epoch + 1)

            if (args.dynamic and args.use_isw and epoch % (args.cov_stat_epoch + 1) == args.cov_stat_epoch) \
               or (args.dynamic is False and args.use_isw and epoch == args.cov_stat_epoch):
                net.module.reset_mask_matrix()
                for trial in range(args.trials):
                    for dataset, val_loader in covstat_val_loaders.items():  # For get the statistics of covariance
                        validate_for_cov_stat(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i,
                                              save_pth=False)
                        net.module.set_mask_matrix()

            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

            if args.class_uniform_pct:
                if epoch >= args.max_cu_epoch:
                    train_obj.build_epoch(cut=True)
                    # train_loader.sampler.set_num_samples()
                    for index in len(train_loaders):
                        train_loaders[index].sampler.set_num_samples()
                else:
                    train_obj.build_epoch()
            
            if epoch % 4 == 3:
                for dataset, val_loader in val_loaders.items():
                    if dataset == "Shift_" + args.weather:
                        current_acc = validate_during_train(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)
                        if current_acc > best_acc:
                            best_acc = current_acc
                            flag = 0
                            save_best_acc(args, net, optim, scheduler, epoch, best_acc)
                        else: # 自定义epoch间隔内，val的精度没有增长，那么终止训练
                            flag = flag + 1
                            break
                if flag == 100:
                    break

            epoch += 1

        ########## 测试模块
        if len(val_loaders) == 1 or (len(val_loaders) != 1 and "shift" == args.dataset[0]):
            # Run validation only one time - To save models
            for dataset, val_loader in val_loaders.items():
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
        else:
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

        for dataset, val_loader in extra_val_loaders.items():
            print("Extra validating... This won't save pth file")
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)
    
    ################################################################### 微调
    elif args.mode == "fine-tune":
        ########## 装载权重模块
        if args.snapshot:
            epoch, mean_iu = optimizer.load_weights_adapter(net, optim, scheduler,
                                args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loaders)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

        ########## 训练模块
        print("#### iteration", i)
        torch.cuda.empty_cache()

        max_iter = args.max_iter*args.num_domains if args.dsbn else args.max_iter

        for train_loader_cnt in range(len(train_loaders)):
            train_dataset_name = train_dataset_names[train_loader_cnt]
            train_loaders_items = [train_loaders[train_loader_cnt]]
            
            i = 0
            epoch = 0
            optim, scheduler = optimizer.get_optimizer(args, net)
            best_acc = 0
            while i < max_iter:
                # Update EPOCH CTR
                cfg.immutable(False)
                cfg.ITER = i
                cfg.immutable(True)

                i = train(train_loaders_items, net, optim, epoch, writer, scheduler, max_iter, train_loader_cnt)
                # 只有多卡训练模式下有用，以使shuffle操作能够在多个epoch中正常工作。否则，dataloader迭代器产生的数据将始终使用相同的顺序（shuffle失效）。
                # train_loader.sampler.set_epoch(epoch + 1)
                # for index in range(len(train_loaders)):
                #     train_loaders[index].sampler.set_epoch(epoch + 1)

                if (args.dynamic and args.use_isw and epoch % (args.cov_stat_epoch + 1) == args.cov_stat_epoch) \
                or (args.dynamic is False and args.use_isw and epoch == args.cov_stat_epoch):
                    net.module.reset_mask_matrix()
                    for trial in range(args.trials):
                        for dataset, val_loader in covstat_val_loaders.items():  # For get the statistics of covariance
                            validate_for_cov_stat(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i,
                                                save_pth=False)
                            net.module.set_mask_matrix()

                if args.local_rank == 0:
                    print("Saving pth file...")
                    evaluate_eval(args, net, optim, scheduler, None, None, [],
                                writer, epoch, "None", None, i, save_pth=True)

                if args.class_uniform_pct:
                    if epoch >= args.max_cu_epoch:
                        train_obj.build_epoch(cut=True)
                        # train_loader.sampler.set_num_samples()
                        for index in len(train_loaders):
                            train_loaders[index].sampler.set_num_samples()
                    else:
                        train_obj.build_epoch()

                if epoch % 4 == 3:
                    for dataset, val_loader in val_loaders.items():
                        if dataset == train_dataset_name:
                            current_acc = validate_during_train(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False, train_loader_cnt=train_loader_cnt)
                            if current_acc > best_acc:
                                best_acc = current_acc
                                flag = 0
                                save_best_acc(args, net, optim, scheduler, epoch, best_acc, train_dataset_name)
                            else: # 自定义epoch间隔内，val的精度没有增长，那么终止训练
                                flag = flag + 1
                                break
                    if flag == 5:
                        break

                epoch += 1

            ########## 测试模块
            if len(val_loaders) == 1 or (len(val_loaders) != 1 and "shift" == args.dataset[0]):
                # Run validation only one time - To save models
                for dataset, val_loader in val_loaders.items():
                    validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, train_loader_cnt=train_loader_cnt)
            else:
                if args.local_rank == 0:
                    print("Saving pth file...")
                    evaluate_eval(args, net, optim, scheduler, None, None, [],
                                writer, epoch, "None", None, i, save_pth=True)

            for dataset, val_loader in extra_val_loaders.items():
                print("Extra validating... This won't save pth file")
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False, train_loader_cnt=train_loader_cnt)
    
    ################################################################### 测试
    elif args.mode == "test":
        ########## 装载权重模块
        if args.snapshot:
            epoch, mean_iu = optimizer.load_weights_adapter(net, optim, scheduler,
                                args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loaders)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

        ########## 测试模块
        print("#### iteration", i)
        torch.cuda.empty_cache()

        if len(val_loaders) == 1 or (len(val_loaders) != 1 and "shift" == args.dataset[0]):
            # Run validation only one time - To save models
            for dataset, val_loader in val_loaders.items():
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
        else:
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

        for dataset, val_loader in extra_val_loaders.items():
            print("Extra validating... This won't save pth file")
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)

    ################################################################### 测试时适应
    elif args.mode == "test-adaptation":
        ########## 装载权重模块
        if args.snapshot:
            epoch, mean_iu = optimizer.load_weights_adapter(net, optim, scheduler,
                                args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loaders)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

        ########## 测试时适应模块
        torch.cuda.empty_cache()

        env_cnt = 0
        results = ['building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 'sidewalk',
                   'vegatation', 'vehicle', 'wall', 'traffic sign', 'sky', 'traffic light', 'terrain', 'miou', 'dataset name']
        
        if args.method == "CoTTA":
            net.copy_state()
            if len(val_loaders) == 1 or (len(val_loaders) != 1 and "shift" == args.dataset[0]):
                # Run validation only one time - To save models
                for dataset, val_loader in val_loaders.items():
                    _, result, iou_cur = test_adaptation_CoTTA(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, train_loader_cnt=0, method=args.method)
                    result.append(dataset)
                    results = np.vstack((results, result))
                    if env_cnt == 0:
                        iou_all = iou_cur
                        env_cnt = env_cnt + 1
                    else:
                        iou_all += iou_cur
                        env_cnt = env_cnt + 1
                    # print(results)
                _, _ = evaluate_eval_all(args, net, optim, scheduler, iou_all, writer, 0, dataset, None, 0)
                print(results)
        
        elif args.method == "ours":
            if len(val_loaders) == 1 or (len(val_loaders) != 1 and "shift" == args.dataset[0]):
                # Run validation only one time - To save models
                for dataset, val_loader in val_loaders.items():
                    _, result, iou_cur = test_adaptation_ours(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, train_loader_cnt=0, method=args.method)
                    result.append(dataset)
                    results = np.vstack((results, result))
                    if env_cnt == 0:
                        iou_all = iou_cur
                        env_cnt = env_cnt + 1
                    else:
                        iou_all += iou_cur
                        env_cnt = env_cnt + 1
                    # print(results)
                _, _ = evaluate_eval_all(args, net, optim, scheduler, iou_all, writer, 0, dataset, None, 0)
                print(results)

        else:
            if len(val_loaders) == 1 or (len(val_loaders) != 1 and "shift" == args.dataset[0]):
                # Run validation only one time - To save models
                for dataset, val_loader in val_loaders.items():
                    _, result, iou_cur = test_adaptation(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, train_loader_cnt=0, method=args.method)
                    result.append(dataset)
                    results = np.vstack((results, result))
                    if env_cnt == 0:
                        iou_all = iou_cur
                        env_cnt = env_cnt + 1
                    else:
                        iou_all += iou_cur
                        env_cnt = env_cnt + 1
                    # print(results)
                    ##### reset #####
                    if args.reset:
                        print("resetting the model")
                        epoch, mean_iu = optimizer.load_weights_adapter(net, optim, scheduler,
                                            args.snapshot, args.restore_optimizer)
                _, _ = evaluate_eval_all(args, net, optim, scheduler, iou_all, writer, 0, dataset, None, 0)
                print(results)


def test_adaptation_CoTTA(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True, train_loader_cnt=0, method="source"):
    # only for CoTTA
    net.train()

    net.requires_grad_(False)

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)

    # 测试时适应模块
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []
    tmp_for_save = ['image_idx', 'image_name', 'building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 'sidewalk', 'vegatation', 'vehicle', 'wall', 'traffic sign', 'sky', 'traffic light', 'terrain', 'miou']

    # resize_func_input = Resize([1, 3, args.crop_size, 1242])
    # resize_func_gt = Resize([1, args.crop_size, 1242])

    for val_idx, data in enumerate(val_loader): # 一张一张来的
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data
        
        gt_image_array = gt_image.numpy()
        
        # class_list = []
        # for i in range(gt_image_array.shape[1]):
        #     for j in range(gt_image_array.shape[2]):
        #         if gt_image_array[0][i][j] not in class_list:
        #             class_list.append(gt_image_array[0][i][j])
        # print (class_list)

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        # Resize

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda() # 必须要手动，使数据在GPU上进行运算

        domain_label = 0 * torch.ones(inputs.shape[0], dtype=torch.long) # 暂时没用
        
        task = train_loader_cnt
        # task = 2

        ##### 前向推理 #####
        optim.zero_grad()
        test_loss, output, style_code = net(inputs, gts=gt_cuda, dsbn=args.dsbn, mode='val', domain_label=domain_label, task=task, run_mode=args.mode, method=method) # 推理

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes
        
        ##### iou计算 #####
        predictions = output.data.max(1)[1].cpu()

        ###### Image Dumps #####
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        num_classes = datasets.num_classes

        iou_current = fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), num_classes)
        if val_idx == 0:
            iou_acc = iou_current
        else:
            iou_acc = iou_acc + iou_current
        # iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                      num_classes)

        ##### inference then adaptation #####
        if net.training:
            # loss计算
            outputs_index = 0
            main_loss = test_loss[outputs_index]
            outputs_index += 1
            aux_loss = test_loss[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            # 根据loss梯度更新
            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            total_loss.backward()
            optim.step()
            # optim.zero_grad()

            del total_loss, log_total_loss

            if method == "CoTTA":
                # Teacher update
                net.teacher_update()
                # Stochastic restore
                net.restore()

        ###### 记录样本深度特征统计量 #####
        if val_idx == 0:
            tmp_style_for_save = [val_idx, img_names]
            for cc in style_code:
                tmp_style_for_save.append(cc)
        else:
            tmp_style = [val_idx, img_names]
            for cc in style_code:
                tmp_style.append(cc)
            tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_style))
            
            del tmp_style
            
        del style_code

        ###### 记录逐样本精度 #####
        iu, mean_iu = record_sample(val_idx, img_names, iou_current)
        tmp = [val_idx, img_names]
        for idx in range(num_classes):
            tmp.append(iu[idx])
        tmp.append(mean_iu)
        tmp_for_save = np.vstack((tmp_for_save, tmp))
        
        ##### Logging #####
        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        if val_idx % 30 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d, lr: %f, cur_iou: %f", val_idx + 1, len(val_loader), optim.param_groups[-1]['lr'], mean_iu)
        if val_idx > 10 and args.test_mode:
            break
        
        del output, val_idx, data, iou_current, iu, mean_iu, tmp, inputs, gt_image, gt_cuda

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()
    
    if args.local_rank == 0:
        iu, mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    # 计算风格平均值
    avg = {}
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = 0

    for row in range(tmp_style_for_save.shape[0]):
        for col in range(tmp_style_for_save.shape[1]-2):
            avg[str(col)] = avg[str(col)] + tmp_style_for_save[row][col+2]
    
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = avg[str(col)] / tmp_style_for_save.shape[0]

    tmp_avg = ['', '']
    for idx in range(len(avg)):
        tmp_avg.append(avg[str(idx)])
    tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_avg))

    # 记录测试集总精度
    tmp = ['total', 'total']
    result = []
    for idx in range(num_classes):
        tmp.append(iu[idx])
        result.append(iu[idx])
    tmp.append(mean_iu)
    result.append(mean_iu)
    tmp_for_save = np.vstack((tmp_for_save, tmp))

    # 保存记录风格
    # tmp_style_for_save = pd.DataFrame(tmp_style_for_save)
    # __writer = pd.ExcelWriter('./rfnet_31_Research_CampusE1_style_split_all.xlsx')
    # tmp_style_for_save.to_excel(__writer, float_format='%.4f', index=False, header=False)
    # __writer.save()
    # __writer.close()
    
    # 保存记录精度
    tmp_for_save = pd.DataFrame(tmp_for_save)
    _writer = pd.ExcelWriter('./' + args.arch.split('.')[-1] + '_Shift_' + args.weather + '_inf.xlsx') # 写入Excel文件
    tmp_for_save.to_excel(_writer,  float_format='%.4f', index=False, header=False)		
    _writer.save()
    _writer.close()

    return val_loss.avg, result, iou_acc

def test_adaptation_ours(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True, train_loader_cnt=0, method="source"):
    # only for ours
    net.train()

    ############### 冻结骨干网络，只更新adapters ###############
    print("################## begin freezing #################")
    # 冻结前两层（深度特征统计量提取层）
    freeze_paras = ['backbone.conv1.weight', 'backbone.bn1.weight', 'backbone.bn1.bias']

    for name, para in net.named_parameters():
        if name in freeze_paras or 'backbone.layer1' in name:
            # print(name)
            para.requires_grad_(False)
        """
        if name.find('bn') != -1:
            if '64' in str(para.shape):
                print(name)
        """

    net.apply(freeze_bn)
        
    global freeze_bn_num
    freeze_bn_num = 0

    # 冻结骨干网络的conv
    for name, m in net.named_modules():
        # Conv
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            m.weight.requires_grad = False
            # print(name)
        if isinstance(m, nn.Conv2d) and 'downsample' in name:
            m.weight.requires_grad = False
            # print(name)
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==1) and ('spp' in name or 'upsample' in name) and ('parallel_conv' not in name):
            m.weight.requires_grad = False
            # print(name)
        # BN
        if isinstance(m, nn.BatchNorm2d) and ('downsample' in name):
            m.weight.requires_grad = False
            m.eval()
            # print(name)

    # 测试时适应模块
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []
    tmp_for_save = ['image_idx', 'image_name', 'building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 'sidewalk', 'vegatation', 'vehicle', 'wall', 'traffic sign', 'sky', 'traffic light', 'terrain', 'miou']

    # resize_func_input = Resize([1, 3, args.crop_size, 1242])
    # resize_func_gt = Resize([1, args.crop_size, 1242])

    for val_idx, data in enumerate(val_loader): # 一张一张来的
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data
        
        gt_image_array = gt_image.numpy()
        
        # class_list = []
        # for i in range(gt_image_array.shape[1]):
        #     for j in range(gt_image_array.shape[2]):
        #         if gt_image_array[0][i][j] not in class_list:
        #             class_list.append(gt_image_array[0][i][j])
        # print (class_list)

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        # Resize

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda() # 必须要手动，使数据在GPU上进行运算

        domain_label = 0 * torch.ones(inputs.shape[0], dtype=torch.long) # 暂时没用
        
        task = train_loader_cnt
        # task = 2

        ##### 前向推理 #####
        optim.zero_grad()
        test_loss, output, style_code = net(inputs, gts=gt_cuda, dsbn=args.dsbn, mode='val', domain_label=domain_label, task=task, run_mode=args.mode, method=method) # 推理

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes
        
        ##### iou计算 #####
        predictions = output.data.max(1)[1].cpu()

        ###### Image Dumps #####
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        num_classes = datasets.num_classes

        iou_current = fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), num_classes)
        if val_idx == 0:
            iou_acc = iou_current
        else:
            iou_acc = iou_acc + iou_current
        # iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                      num_classes)

        ##### inference then adaptation #####
        if net.training:
            # loss计算
            outputs_index = 0
            main_loss = test_loss[outputs_index]
            outputs_index += 1
            aux_loss = test_loss[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            # 根据loss梯度更新
            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            total_loss.backward()
            optim.step()
            # optim.zero_grad()

            del total_loss, log_total_loss

        ###### 记录样本深度特征统计量 #####
        if val_idx == 0:
            tmp_style_for_save = [val_idx, img_names]
            for cc in style_code:
                tmp_style_for_save.append(cc)
        else:
            tmp_style = [val_idx, img_names]
            for cc in style_code:
                tmp_style.append(cc)
            tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_style))
            
            del tmp_style
            
        del style_code

        ###### 记录逐样本精度 #####
        iu, mean_iu = record_sample(val_idx, img_names, iou_current)
        tmp = [val_idx, img_names]
        for idx in range(num_classes):
            tmp.append(iu[idx])
        tmp.append(mean_iu)
        tmp_for_save = np.vstack((tmp_for_save, tmp))
        
        ##### Logging #####
        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        if val_idx % 30 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d, lr: %f, cur_iou: %f", val_idx + 1, len(val_loader), optim.param_groups[-1]['lr'], mean_iu)
        if val_idx > 10 and args.test_mode:
            break
        
        del output, val_idx, data, iou_current, iu, mean_iu, tmp, inputs, gt_image, gt_cuda

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()
    
    if args.local_rank == 0:
        iu, mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    # 计算风格平均值
    avg = {}
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = 0

    for row in range(tmp_style_for_save.shape[0]):
        for col in range(tmp_style_for_save.shape[1]-2):
            avg[str(col)] = avg[str(col)] + tmp_style_for_save[row][col+2]
    
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = avg[str(col)] / tmp_style_for_save.shape[0]

    tmp_avg = ['', '']
    for idx in range(len(avg)):
        tmp_avg.append(avg[str(idx)])
    tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_avg))

    # 记录测试集总精度
    tmp = ['total', 'total']
    result = []
    for idx in range(num_classes):
        tmp.append(iu[idx])
        result.append(iu[idx])
    tmp.append(mean_iu)
    result.append(mean_iu)
    tmp_for_save = np.vstack((tmp_for_save, tmp))

    # 保存记录风格
    # tmp_style_for_save = pd.DataFrame(tmp_style_for_save)
    # __writer = pd.ExcelWriter('./rfnet_31_Research_CampusE1_style_split_all.xlsx')
    # tmp_style_for_save.to_excel(__writer, float_format='%.4f', index=False, header=False)
    # __writer.save()
    # __writer.close()
    
    # 保存记录精度
    tmp_for_save = pd.DataFrame(tmp_for_save)
    _writer = pd.ExcelWriter('./' + args.arch.split('.')[-1] + '_Shift_' + args.weather + '_inf.xlsx') # 写入Excel文件
    tmp_for_save.to_excel(_writer,  float_format='%.4f', index=False, header=False)		
    _writer.save()
    _writer.close()

    return val_loss.avg, result, iou_acc

def test_adaptation(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True, train_loader_cnt=0, method="source"):
    # 根据method选取测试时适应的模式
    if method == "source":
        net.eval() # 切换至验证模式，仅对Dropout或BatchNorm等模块有任何影响

    elif method == "BN_Adapt":
        net.eval()
      
        # 让BN statistics变化
        for name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.train()
                # m.eps = 1e-5
                # m.momentum = 0.1
                # print(name)
        
    elif method == "Tent":
        net.train()

        net.requires_grad_(False)

        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    elif "PL" in method:
        net.train()

    # 测试时适应模块
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []
    tmp_for_save = ['image_idx', 'image_name', 'building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 'sidewalk', 'vegatation', 'vehicle', 'wall', 'traffic sign', 'sky', 'traffic light', 'terrain', 'miou']

    # resize_func_input = Resize([1, 3, args.crop_size, 1242])
    # resize_func_gt = Resize([1, args.crop_size, 1242])

    for val_idx, data in enumerate(val_loader): # 一张一张来的
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data
        
        gt_image_array = gt_image.numpy()
        
        # class_list = []
        # for i in range(gt_image_array.shape[1]):
        #     for j in range(gt_image_array.shape[2]):
        #         if gt_image_array[0][i][j] not in class_list:
        #             class_list.append(gt_image_array[0][i][j])
        # print (class_list)

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        # Resize

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda() # 必须要手动，使数据在GPU上进行运算

        domain_label = 0 * torch.ones(inputs.shape[0], dtype=torch.long) # 暂时没用
        
        task = train_loader_cnt
        # task = 2

        ##### 前向推理 #####
        optim.zero_grad()
        test_loss, output, style_code = net(inputs, gts=gt_cuda, dsbn=args.dsbn, mode='val', domain_label=domain_label, task=task, run_mode=args.mode, method=method) # 推理

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes
        
        ##### iou计算 #####
        predictions = output.data.max(1)[1].cpu()

        ###### Image Dumps #####
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        num_classes = datasets.num_classes

        iou_current = fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), num_classes)
        if val_idx == 0:
            iou_acc = iou_current
        else:
            iou_acc = iou_acc + iou_current
        # iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                      num_classes)

        ##### inference then adaptation #####
        if net.training:
            # loss计算
            outputs_index = 0
            main_loss = test_loss[outputs_index]
            outputs_index += 1
            aux_loss = test_loss[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            # 根据loss梯度更新
            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            total_loss.backward()
            optim.step()
            # optim.zero_grad()

            del total_loss, log_total_loss

        ###### 记录样本深度特征统计量 #####
        if val_idx == 0:
            tmp_style_for_save = [val_idx, img_names]
            for cc in style_code:
                tmp_style_for_save.append(cc)
        else:
            tmp_style = [val_idx, img_names]
            for cc in style_code:
                tmp_style.append(cc)
            tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_style))
            
            del tmp_style
            
        del style_code

        ###### 记录逐样本精度 #####
        iu, mean_iu = record_sample(val_idx, img_names, iou_current)
        tmp = [val_idx, img_names]
        for idx in range(num_classes):
            tmp.append(iu[idx])
        tmp.append(mean_iu)
        tmp_for_save = np.vstack((tmp_for_save, tmp))
        
        ##### Logging #####
        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        if val_idx % 30 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d, lr: %f, cur_iou: %f", val_idx + 1, len(val_loader), optim.param_groups[-1]['lr'], mean_iu)
        if val_idx > 10 and args.test_mode:
            break
        
        del output, val_idx, data, iou_current, iu, mean_iu, tmp, inputs, gt_image, gt_cuda

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()
    
    if args.local_rank == 0:
        iu, mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    # 计算风格平均值
    avg = {}
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = 0

    for row in range(tmp_style_for_save.shape[0]):
        for col in range(tmp_style_for_save.shape[1]-2):
            avg[str(col)] = avg[str(col)] + tmp_style_for_save[row][col+2]
    
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = avg[str(col)] / tmp_style_for_save.shape[0]

    tmp_avg = ['', '']
    for idx in range(len(avg)):
        tmp_avg.append(avg[str(idx)])
    tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_avg))

    # 记录测试集总精度
    tmp = ['total', 'total']
    result = []
    for idx in range(num_classes):
        tmp.append(iu[idx])
        result.append(iu[idx])
    tmp.append(mean_iu)
    result.append(mean_iu)
    tmp_for_save = np.vstack((tmp_for_save, tmp))

    # 保存记录风格
    # tmp_style_for_save = pd.DataFrame(tmp_style_for_save)
    # __writer = pd.ExcelWriter('./rfnet_31_Research_CampusE1_style_split_all.xlsx')
    # tmp_style_for_save.to_excel(__writer, float_format='%.4f', index=False, header=False)
    # __writer.save()
    # __writer.close()
    
    # 保存记录精度
    tmp_for_save = pd.DataFrame(tmp_for_save)
    _writer = pd.ExcelWriter('./' + args.arch.split('.')[-1] + '_Shift_' + args.weather + '_inf.xlsx') # 写入Excel文件
    tmp_for_save.to_excel(_writer,  float_format='%.4f', index=False, header=False)		
    _writer.save()
    _writer.close()

    return val_loss.avg, result, iou_acc


def freeze_bn(m):
    global freeze_bn_num
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.num_features == 64 and freeze_bn_num < 10:
            # print(classname)
            m.eval()
            freeze_bn_num = freeze_bn_num + 1

def train(train_loaders, net, optim, curr_epoch, writer, scheduler, max_iter, train_loader_cnt):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()


    ############### 冻结骨干网络，只更新adapters ###############
    print("################## begin freezing #################")
    # 冻结前两层（深度特征统计量提取层）
    freeze_paras = ['backbone.conv1.weight', 'backbone.bn1.weight', 'backbone.bn1.bias']

    for name, para in net.named_parameters():
        if name in freeze_paras or 'backbone.layer1' in name:
            # print(name)
            para.requires_grad_(False)
        """
        if name.find('bn') != -1:
            if '64' in str(para.shape):
                print(name)
        """

    net.apply(freeze_bn)
    
    global freeze_bn_num
    freeze_bn_num = 0

    # 冻结骨干网络的conv
    for name, m in net.named_modules():
        # Conv
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            m.weight.requires_grad = False
            # print(name)
        if isinstance(m, nn.Conv2d) and 'downsample' in name:
            m.weight.requires_grad = False
            # print(name)
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==1) and ('spp' in name or 'upsample' in name) and ('parallel_conv' not in name):
            m.weight.requires_grad = False
            # print(name)
        # BN
        if isinstance(m, nn.BatchNorm2d) and ('downsample' in name):
            m.weight.requires_grad = False
            m.eval()
            # print(name)


    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    # len(train_loaders) = 训练所用数据集个数；len(train_loaders[0]) = 一个训练数据集包含的样本batch数
    # curr_iter = curr_epoch * len(train_loader) —— DSBN所用的不同数据集具有相同结构和数量的数据（源图像的不同copy）
    curr_iter = curr_epoch * len(train_loaders[0]) * len(train_loaders)
    # print (len(train_loaders[0]))
    # print (max_iter)

    # train_loaders的迭代器 —— enumerate(DataLoader)返回的就是DataLoader对应数据集Class的__getitem__()函数的内容的list（再进行一个list[]操作）！
    train_loader_iters = [enumerate(train_loader) for train_loader in train_loaders]

    # 一个epoch的训练(注意：由于共享索引i，其实一个epoch包含了train_loaders[0] * dataset个iterations；一个i索引也就包含了dataset个iterations)
    for i in range(len(train_loaders[0])):
        if curr_iter >= max_iter:
            break
        inputs = [] # inputs[0]存放第一个数据集的一个输入；inputs[1]存放第二个……
        gts = []
        aux_gts = []
        for dataset_index in range(len(train_loader_iters)):
            _, (input, gt, _, aux_gt) = train_loader_iters[dataset_index].__next__()
            inputs.append(input)
            gts.append(gt)
            aux_gts.append(aux_gt)

        B, C, H, W = inputs[0].shape
        batch_pixel_size = C * H * W

        # if H != 768:
        #     print('aaaasdasd')

        # print(i)
        # 主体训练部分
        for dataset_index in range(len(train_loader_iters)):
            img_gt = None
            input, gt, aux_gt = inputs[dataset_index].cuda(), gts[dataset_index].cuda(), aux_gts[dataset_index].cuda()

            # print(input.size())
            start_ts = time.time()
            
            domain_label = dataset_index * torch.ones(input.shape[0], dtype=torch.long) # 暂无用处

            task = train_loader_cnt
            # task = 2

            # 前传与梯度计算
            optim.zero_grad()
            if args.use_isw:
                outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature,
                            apply_wtloss=False if curr_epoch<=args.cov_stat_epoch else True, dsbn=args.dsbn, mode='train', domain_label=domain_label)
            else:
                outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature, dsbn=args.dsbn, mode='train', domain_label=domain_label, task=task)
            
            # loss计算
            outputs_index = 0
            main_loss = outputs[outputs_index]
            outputs_index += 1
            aux_loss = outputs[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            if args.use_wtloss and (not args.use_isw or (args.use_isw and curr_epoch > args.cov_stat_epoch)):
                wt_loss = outputs[outputs_index]
                outputs_index += 1
                total_loss = total_loss + (args.wt_reg_weight * wt_loss)
            else:
                wt_loss = 0

            if args.visualize_feature:
                f_cor_arr = outputs[outputs_index]
                outputs_index += 1
            
            # 根据loss梯度更新
            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)
            total_loss.backward()
            optim.step()

            del total_loss, log_total_loss

            # 可视化
            time_meter.update((time.time() - start_ts))

            if args.local_rank == 0:
                if i % 50 == 49:
                    if args.visualize_feature:
                        visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [for dataset_index: {}], [lr {:0.6f}], [time {:0.4f}], [task {}]'.format(
                        curr_epoch, curr_iter % (curr_epoch * len(train_loaders[0]) * len(train_loaders)) + 1 if curr_epoch > 0 else curr_iter + 1,
                        len(train_loaders) * len(train_loaders[0]), curr_iter + 1, train_total_loss.avg, dataset_index,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size, task)

                    logging.info(msg)
                    if args.use_wtloss:
                        print("Whitening Loss", wt_loss)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

            # print(curr_iter, max_iter)
            curr_iter += 1
            scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter


    """ 原
    for i, data in enumerate(train_loaders[0]):
        # print("i: %d" %(i))
        if curr_iter >= max_iter:
            break

        inputs, gts, _, aux_gts = data

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            # print("di: %d" %(di))
            input, gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None
            input, gt = input.cuda(), gt.cuda()

            optim.zero_grad()
            if args.use_isw:
                outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature,
                            apply_wtloss=False if curr_epoch<=args.cov_stat_epoch else True)
            else:
                outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)
            outputs_index = 0
            main_loss = outputs[outputs_index]
            outputs_index += 1
            aux_loss = outputs[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            if args.use_wtloss and (not args.use_isw or (args.use_isw and curr_epoch > args.cov_stat_epoch)):
                wt_loss = outputs[outputs_index]
                outputs_index += 1
                total_loss = total_loss + (args.wt_reg_weight * wt_loss)
            else:
                wt_loss = 0

            if args.visualize_feature:
                f_cor_arr = outputs[outputs_index]
                outputs_index += 1

            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss

            if args.local_rank == 0:
                if i % 50 == 49:
                    if args.visualize_feature:
                        visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loaders[0]), curr_iter, train_total_loss.avg,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                    logging.info(msg)
                    if args.use_wtloss:
                        print("Whitening Loss", wt_loss)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter
    """


def validate_during_train(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True, train_loader_cnt=0):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """
    net.eval() # 切换至验证模式，仅对Dropout或BatchNorm等模块有任何影响
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []
    
    # resize_func_input = Resize([1, 3, args.crop_size, args.crop_size])
    # resize_func_input = Resize([1, 3, args.crop_size, args.crop_size])

    for val_idx, data in enumerate(val_loader): # 一张一张来的
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        # Resize

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda() # 必须要手动，使数据在GPU上进行运算

        domain_label = 0 * torch.ones(inputs.shape[0], dtype=torch.long)

        task = train_loader_cnt

        with torch.no_grad():
            if args.use_wtloss:
                output, f_cor_arr = net(inputs, visualize=True, dsbn=args.dsbn, mode='val', domain_label=domain_label)
            else:
                output, style_code = net(inputs, dsbn=args.dsbn, mode='val', domain_label=domain_label, task=task) # 推理

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        if "campuse1" == args.dataset[0]:
            assert output.size()[1] == 10
        elif "new_campuse1" == args.dataset[0]:
            assert output.size()[1] == 31
        else:
            assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 30 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        if "campuse1" == args.dataset[0]:
            num_classes = 10
        elif "new_campuse1" == args.dataset[0]:
            num_classes = 31
        else:
            num_classes = datasets.num_classes

        iou_current = fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), num_classes)
        if val_idx == 0:
            iou_acc = iou_current
        else:
            iou_acc = iou_acc + iou_current
        # iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                      num_classes)
        
        del output, val_idx, data, iou_current

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()
    
    if args.local_rank == 0:
        iu, mean_iu = evaluate_eval_dur_train(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

        if args.use_wtloss:
            visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    return mean_iu


def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True, train_loader_cnt=0):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval() # 切换至验证模式，仅对Dropout或BatchNorm等模块有任何影响
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []
    tmp_for_save = ['image_idx', 'image_name', 'building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 'sidewalk', 'vegatation', 'vehicle', 'wall', 'traffic sign', 'sky', 'traffic light', 'terrain', 'miou']

    # resize_func_input = Resize([1, 3, args.crop_size, 1242])
    # resize_func_gt = Resize([1, args.crop_size, 1242])

    for val_idx, data in enumerate(val_loader): # 一张一张来的
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data
        
        gt_image_array = gt_image.numpy()
        
        # class_list = []
        # for i in range(gt_image_array.shape[1]):
        #     for j in range(gt_image_array.shape[2]):
        #         if gt_image_array[0][i][j] not in class_list:
        #             class_list.append(gt_image_array[0][i][j])
        # print (class_list)

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        # Resize

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda() # 必须要手动，使数据在GPU上进行运算

        domain_label = 0 * torch.ones(inputs.shape[0], dtype=torch.long) # 暂时没用
        
        task = train_loader_cnt
        # task = 2

        with torch.no_grad():
            if args.use_wtloss:
                output, f_cor_arr = net(inputs, visualize=True, dsbn=args.dsbn, mode='val', domain_label=domain_label)
            else:
                output, style_code = net(inputs, dsbn=args.dsbn, mode='val', domain_label=domain_label, task=task) # 推理

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        if "campuse1" == args.dataset[0]:
            assert output.size()[1] == 10
        elif "new_campuse1" == args.dataset[0]:
            assert output.size()[1] == 31
        else:
            assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 30 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        if "campuse1" == args.dataset[0]:
            num_classes = 10
        elif "new_campuse1" == args.dataset[0]:
            num_classes = 31
        else:
            num_classes = datasets.num_classes

        iou_current = fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), num_classes)
        if val_idx == 0:
            iou_acc = iou_current
        else:
            iou_acc = iou_acc + iou_current
        # iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                      num_classes)

        # 记录样本风格
        if val_idx == 0:
            tmp_style_for_save = [val_idx, img_names]
            for cc in style_code:
                tmp_style_for_save.append(cc)
        else:
            tmp_style = [val_idx, img_names]
            for cc in style_code:
                tmp_style.append(cc)
            tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_style))
            
            del tmp_style
            
        del style_code

        # 记录逐样本精度
        iu, mean_iu = record_sample(val_idx, img_names, iou_current)
        tmp = [val_idx, img_names]
        for idx in range(num_classes):
            tmp.append(iu[idx])
        tmp.append(mean_iu)
        tmp_for_save = np.vstack((tmp_for_save, tmp))
        
        del output, val_idx, data, iou_current, iu, mean_iu, tmp

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()
    
    if args.local_rank == 0:
        iu, mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

        if args.use_wtloss:
            visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    # 计算风格平均值
    avg = {}
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = 0

    for row in range(tmp_style_for_save.shape[0]):
        for col in range(tmp_style_for_save.shape[1]-2):
            avg[str(col)] = avg[str(col)] + tmp_style_for_save[row][col+2]
    
    for col in range(tmp_style_for_save.shape[1]-2):
        avg[str(col)] = avg[str(col)] / tmp_style_for_save.shape[0]

    tmp_avg = ['', '']
    for idx in range(len(avg)):
        tmp_avg.append(avg[str(idx)])
    tmp_style_for_save = np.vstack((tmp_style_for_save, tmp_avg))

    # 记录测试集总精度
    tmp = ['total', 'total']
    for idx in range(num_classes):
        tmp.append(iu[idx])
    tmp.append(mean_iu)
    tmp_for_save = np.vstack((tmp_for_save, tmp))

    # 保存记录风格
    # tmp_style_for_save = pd.DataFrame(tmp_style_for_save)
    # __writer = pd.ExcelWriter('./rfnet_31_Research_CampusE1_style_split_all.xlsx')
    # tmp_style_for_save.to_excel(__writer, float_format='%.4f', index=False, header=False)
    # __writer.save()
    # __writer.close()
    
    # 保存记录精度
    tmp_for_save = pd.DataFrame(tmp_for_save)
    _writer = pd.ExcelWriter('./' + args.arch.split('.')[-1] + '_Shift_' + args.weather + '_inf.xlsx') # 写入Excel文件
    tmp_for_save.to_excel(_writer,  float_format='%.4f', index=False, header=False)		
    _writer.save()
    _writer.close()

    return val_loss.avg

def record_sample(val_idx, img_names, hist):
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    #logging.info("image_id: {}, image_name: {}".format(val_idx, img_names))
    for idx, i in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        iu_string = '{:5.1f}'.format(i * 100)
    #    logging.info("label_id: {}, iU: {}".format(idx_string, iu_string))
        
    mean_iu = np.nanmean(iu)
    #logging.info('mean iU: {}'.format(mean_iu))
    
    return iu, mean_iu

def validate_for_cov_stat(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    # net.train()#eval()
    net.eval()

    for val_idx, data in enumerate(val_loader):
        img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
        img_or, img_photometric = img_or.cuda(), img_photometric.cuda()

        with torch.no_grad():
            net([img_photometric, img_or], cal_covstat=True)

        del img_or, img_photometric, img_geometric

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / 100", val_idx + 1)
        del data

        if val_idx >= 499:
            return


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = '/home/userA/projects/visualization/feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()

# ssh -L 6123:127.0.0.1:6123；tensorboard --logdir="./logs/log_path/" --port=6123