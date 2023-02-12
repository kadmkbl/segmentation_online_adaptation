"""
Dataset setup and loaders
"""
from datasets import cityscapes # 为什么这里还需要from datasets，是因为调用到这里的文件是train.py和valid.py，所以主路径还是在第一级的，而非datasets这个第二级路径上。
from datasets import mapillary
from datasets import synthia
from datasets import kitti
from datasets import camvid
from datasets import bdd100k
from datasets import campus
from datasets import new_campus
from datasets import gtav
from datasets import acdc
from datasets import synthia_seq
from datasets import virtual_kitti2
from datasets import c_driving
from datasets import shift
from datasets import nullloader

from datasets import multi_loader
from datasets.sampler import DistributedSampler

import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch

# 默认公用语义分割设置（cityscapes配置）
num_classes = 14 # 这个和train.py的num_classes影响最后的评估程序；new_campuse1中的num_classes影响label识别；模型的num_classes影响模型结构
ignore_label = 255


def get_train_joint_transform(args, dataset): # 其实也会有一定的随机性，但是epoch轮数高了后这种随机性会下降。
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """

    # Geometric image transformations
    train_joint_transform_list = []
    train_joint_transform_list += [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           crop_nopad=args.crop_nopad,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=dataset.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]

    if args.rrotate > 0:
        train_joint_transform_list += [joint_transforms.RandomRotate(
            degree=args.rrotate,
            ignore_index=dataset.ignore_label)]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # return the raw list for class uniform sampling
    return train_joint_transform_list, train_joint_transform


def get_input_transforms(args, dataset):
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """

    # Image appearance transformations
    train_input_transform = []
    val_input_transform = []
    if args.color_aug > 0.0:
        train_input_transform += [standard_transforms.RandomApply([
            standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]

    train_input_transform += [
                                standard_transforms.ToTensor()
    ]
    val_input_transform += [
                            standard_transforms.ToTensor()
    ]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    val_input_transform = standard_transforms.Compose(val_input_transform)

    return train_input_transform, val_input_transform

def get_color_geometric_transforms():
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """

    # Image appearance transformations
    color_input_transform = []
    geometric_input_transform = []

    color_input_transform += [standard_transforms.ColorJitter(0.8, 0.8, 0.8, 0.3)]
    color_input_transform += [extended_transforms.RandomGaussianBlur()]

    geometric_input_transform += [standard_transforms.RandomHorizontalFlip(p=1.0)]

    color_input_transform += [
                              standard_transforms.ToTensor()
    ]
    geometric_input_transform += [
                            standard_transforms.ToTensor()
    ]
    color_input_transform = standard_transforms.Compose(color_input_transform)
    geometric_input_transform = standard_transforms.Compose(geometric_input_transform)

    return color_input_transform, geometric_input_transform

def get_target_transforms(args, dataset):
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """

    target_transform = extended_transforms.MaskToTensor()
    if args.jointwtborder:
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
                dataset.ignore_label, dataset.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    target_aux_train_transform = extended_transforms.MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform


def create_extra_val_loader(args, dataset, val_input_transform, target_transform, val_sampler):
    """
    Create extra validation loader
    Args:
        args: input config arguments
        dataset: dataset class object
        val_input_transform: validation input transforms
        target_transform: target transforms
        val_sampler: validation sampler

    return: validation loaders
    """
    if dataset == 'cityscapes':
        val_set = cityscapes.CityScapes('fine', 'val', 0,
                                        transform=val_input_transform,
                                        target_transform=target_transform,
                                        cv_split=args.cv,
                                        image_in=args.image_in)
    elif dataset == 'bdd100k':
        val_set = bdd100k.BDD100K('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
    elif dataset == 'gtav':
        val_set = gtav.GTAV('val', 0,
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in)
    elif dataset == 'synthia':
        val_set = synthia.Synthia('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
    elif dataset == 'mapillary':
        eval_size = 1536
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        val_set = mapillary.Mapillary('semantic', 'val',
                                      joint_transform_list=val_joint_transform_list,
                                      transform=val_input_transform,
                                      target_transform=target_transform,
                                      test=False)
    elif dataset == 'acdc':
        val_set = acdc.ACDC('val', args.weather, 0, 
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in)
    elif dataset == 'campuse1':
        val_set = campus.CampusE1('val', 0,
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in)
    elif dataset == 'null_loader':
        val_set = nullloader.nullloader(args.crop_size)
    else:
        raise Exception('Dataset {} is not supported'.format(dataset))

    if args.syncbn:
        from datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                            sampler = val_sampler)
    return val_loader

def create_covstat_val_loader(args, dataset, val_input_transform, target_transform, val_sampler):
    """
    Create covariance statistics validation loader
    Args:
        args: input config arguments
        dataset: dataset class object
        val_input_transform: validation input transforms
        target_transform: target transforms
        val_sampler: validation sampler

    return: validation loaders
    """

    color_transform, geometric_transform = get_color_geometric_transforms()
    if dataset == 'cityscapes':
        val_set = cityscapes.CityScapesAug('fine', 'train', 0,
                                        transform=val_input_transform,
                                        color_transform=color_transform,
                                        geometric_transform=geometric_transform,
                                        target_transform=target_transform,
                                        cv_split=args.cv,
                                        image_in=args.image_in)
    elif dataset == 'bdd100k':
        val_set = bdd100k.BDD100KAug('train', 0,
                                  transform=val_input_transform,
                                  color_transform=color_transform,
                                  geometric_transform=geometric_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
    elif dataset == 'gtav':
        val_set = gtav.GTAVAug('train', 0,
                            transform=val_input_transform,
                            color_transform=color_transform,
                            geometric_transform=geometric_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in)
    elif dataset == 'synthia':
        val_set = synthia.SynthiaAug('train', 0,
                                  transform=val_input_transform,
                                  color_transform=color_transform,
                                  geometric_transform=geometric_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
    elif dataset == 'mapillary':
        print("Not supported")
        exit()
    elif dataset == 'null_loader':
        val_set = nullloader.nullloader(args.crop_size)
    else:
        raise Exception('Dataset {} is not supported'.format(dataset))

    # if args.syncbn:
    #     from datasets.sampler import DistributedSampler
    #     val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)
    # else:
    val_sampler = None
    val_loader = DataLoader(val_set, batch_size=1,
                            num_workers=args.num_workers // 2 , shuffle=True, drop_last=False,
                            sampler = val_sampler)
    return val_loader

def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, bdd100k, gtav, synthia, mapillary, ACDC]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    args.train_batch_size = args.bs_mult * args.ngpu
    if args.bs_mult_val > 0:
        args.val_batch_size = args.bs_mult_val * args.ngpu
    else:
        args.val_batch_size = args.bs_mult * args.ngpu

    # Readjust batch size to mini-batch size for syncbn
    if args.syncbn:
        args.train_batch_size = args.bs_mult
        args.val_batch_size = args.bs_mult_val

    args.num_workers = 8 #1 * args.ngpu
    if args.test_mode:
        args.num_workers = 1

    train_sets = []
    val_sets = []
    val_dataset_names = []
    train_dataset_names = []

    if 'cityscapes' in args.dataset:
        dataset = cityscapes
        city_mode = args.city_mode #'train' ## Can be trainval
        city_quality = 'fine'
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct: # 0.5 —— What fraction of images is uniformly sampled
            if args.coarse_boost_classes: # use coarse annotations to boost fine data with specific classes
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.CityScapesUniform(
                city_quality, city_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.CityScapes(
                city_quality, city_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in)

        val_set = dataset.CityScapes('fine', 'val', 0,
                                     transform=val_input_transform,
                                     target_transform=target_transform,
                                     cv_split=args.cv,
                                     image_in=args.image_in)

        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('cityscapes')

        if args.dsbn and args.class_uniform_pct:
            del train_sets[0]
            for index in range(args.num_domains):
                train_set = dataset.CityScapesUniform(
                    city_quality, city_mode, args.maxSkip,
                    joint_transform_list=train_joint_transform_list,
                    transform=train_input_transform,
                    target_transform=target_train_transform,
                    target_aux_transform=target_aux_train_transform,
                    dump_images=args.dump_augmentation_images,
                    cv_split=args.cv,
                    class_uniform_pct=args.class_uniform_pct,
                    class_uniform_tile=args.class_uniform_tile,
                    test=args.test_mode,
                    coarse_boost_classes=coarse_boost_classes,
                    image_in=args.image_in,
                    augs=True,
                    domain_index=index)
                train_sets.append(train_set)

    if 'bdd100k' in args.dataset:
        dataset = bdd100k
        bdd_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.BDD100KUniform(
                bdd_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.BDD100K(
                bdd_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in)

        val_set = dataset.BDD100K('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('bdd100k')

    if 'gtav' in args.dataset:
        dataset = gtav
        gtav_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.GTAVUniform(
                gtav_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = gtav.GTAV(
                gtav_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in)

        val_set = gtav.GTAV('val', 0,
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in)
        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('gtav')

    if 'synthia' in args.dataset:
        dataset = synthia
        synthia_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.SynthiaUniform(
                synthia_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.Synthia(
                synthia_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in)

        val_set = dataset.Synthia('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('synthia')

    if 'mapillary' in args.dataset:
        dataset = mapillary
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        eval_size = 1536
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]

        train_set = dataset.Mapillary(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            target_aux_transform=target_aux_train_transform,
            image_in=args.image_in,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode)
        val_set = dataset.Mapillary(
            'semantic', 'val',
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            image_in=args.image_in,
            test=False)
        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('mapillary')

    if 'acdc' in args.dataset:
        dataset = acdc
        acdc_mode = 'train'
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct: # 0.5 —— What fraction of images is uniformly sampled
            if args.coarse_boost_classes: # use coarse annotations to boost fine data with specific classes
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.ACDCUniform(
                acdc_mode, args.weather, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.ACDC(
                acdc_mode, args.weather, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in)

        train_sets.append(train_set)
    
        envs = ['snow', 'rain', 'night', "fog", 'all']
        # envs = ['E1_1F']
        for env in list(envs):
            val_set = dataset.ACDC('val', env, 0,
                                transform=val_input_transform,
                                target_transform=target_transform,
                                cv_split=args.cv,
                                image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('ACDC_' + env)
    
    if 'synthia_seq' in args.dataset:
        dataset = synthia_seq
        synthia_seq_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct: # 0.5 —— What fraction of images is uniformly sampled
            if args.coarse_boost_classes: # use coarse annotations to boost fine data with specific classes
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.STNTHIA_SEQ_Uniform(
                synthia_seq_mode, args.weather, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.STNTHIA_SEQ(
                synthia_seq_mode, args.weather, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in)

        train_sets.append(train_set)
    
        # envs = ['snow', 'rain', 'night', "fog", 'all']
        envs = ['all']
        for env in list(envs):
            val_set = dataset.STNTHIA_SEQ('val', env, 0,
                                transform=val_input_transform,
                                target_transform=target_transform,
                                cv_split=args.cv,
                                image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('Synthia_SEQ_' + env)
            
    if 'virtual_kitti2' in args.dataset:
        dataset = virtual_kitti2
        synthia_seq_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct: # 0.5 —— What fraction of images is uniformly sampled
            if args.coarse_boost_classes: # use coarse annotations to boost fine data with specific classes
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.VIRTUAL_KITTI2_Uniform(
                synthia_seq_mode, args.weather, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.VIRTUAL_KITTI2(
                synthia_seq_mode, args.weather, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in)

        train_sets.append(train_set)

        # envs = ['snow', 'rain', 'night', "fog", 'all']
        envs = ['all']
        for env in list(envs):
            val_set = dataset.VIRTUAL_KITTI2('val', env, 0,
                                transform=val_input_transform,
                                target_transform=target_transform,
                                cv_split=args.cv,
                                image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('Virtual_KITTI2_' + env)

    if 'c_driving' in args.dataset:
        dataset =  c_driving
        synthia_seq_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct: # 0.5 —— What fraction of images is uniformly sampled
            if args.coarse_boost_classes: # use coarse annotations to boost fine data with specific classes
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.C_Driving_Uniform(
                synthia_seq_mode, args.weather, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.C_Driving(
                synthia_seq_mode, args.weather, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in)

        train_sets.append(train_set)

        # envs = ['snow', 'rain', 'night', "fog", 'all']
        envs = ['all']
        for env in list(envs):
            val_set = dataset.C_Driving('val', env, 0,
                                transform=val_input_transform,
                                target_transform=target_transform,
                                cv_split=args.cv,
                                image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('C_Driving_' + env)
            
    if 'shift' in args.dataset:
        dataset = shift
        synthia_seq_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct: # 0.5 —— What fraction of images is uniformly sampled
            if args.coarse_boost_classes: # use coarse annotations to boost fine data with specific classes
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.SHIFT_Uniform(
                synthia_seq_mode, args.weather, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        # else:
            # train_set = dataset.SHIFT(
            #     synthia_seq_mode, args.weather, 0,
            #     joint_transform=train_joint_transform,
            #     transform=train_input_transform,
            #     target_transform=target_train_transform,
            #     target_aux_transform=target_aux_train_transform,
            #     dump_images=args.dump_augmentation_images,
            #     image_in=args.image_in)


        #     envs = ['clear_dawn', 'clear_daytime', 'clear_night']
        #     for env in list(envs):
        #         train_set = dataset.SHIFT(
        #             synthia_seq_mode, env, 0,
        #             transform=train_input_transform,
        #             target_transform=target_train_transform,
        #             target_aux_transform=target_aux_train_transform,
        #             dump_images=args.dump_augmentation_images,
        #             image_in=args.image_in)
        #         train_sets.append(train_set)
        #         train_dataset_names.append('Shift_' + env)

        envs = ['clear_dawn', 'clear_daytime', 'clear_night', 'cloudy_dawn', 'cloudy_daytime', 'cloudy_night',
                'foggy_dawn', 'foggy_daytime', 'foggy_night', 'overcast_dawn', 'overcast_daytime', 'overcast_night',
                'rainy_dawn', 'rainy_daytime', 'rainy_night']
        # envs = ['clear_dawn', 'clear_daytime', 'clear_night', 'all']
        for env in list(envs):
            val_set = dataset.SHIFT('val', env, 0,
                                transform=val_input_transform,
                                target_transform=target_transform,
                                cv_split=args.cv,
                                image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('Shift_' + env)

    if 'campuse1' in args.dataset:
        dataset = campus
        campus_mode = "train"
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.CampusE1Uniform(
                campus_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in,
                split=args.split)
        else:
            train_set = dataset.CampusE1(
                campus_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in,
                split=args.split)

        val_set = dataset.CampusE1('val', 0,
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in,
                            split=args.split)
        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('New_CampusE1')

    if 'new_campuse1' in args.dataset:
        dataset = new_campus
        campus_mode = "train"
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.CampusE1Uniform(
                campus_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in,
                split=args.split)
        else:
            train_set = dataset.CampusE1(
                campus_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in,
                split=args.split)

        train_sets.append(train_set)
        # print(len(train_set))
        
        envs = ['E1G', 'E1_2F', 'E1_1F', "all"]
        # envs = ['E1_1F']
        for env in list(envs):
            val_set = dataset.CampusE1('val', 0,
                                transform=val_input_transform,
                                target_transform=target_transform,
                                cv_split=args.cv,
                                image_in=args.image_in,
                                split=args.split,
                                env=env)
            val_sets.append(val_set)
            val_dataset_names.append('New_CampusE1_' + env)

    if 'null_loader' in args.dataset:
        train_set = nullloader.nullloader(args.crop_size)
        val_set = nullloader.nullloader(args.crop_size)

        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('null_loader')

    if len(train_sets) == 0:
        train_set = 0
        train_loaders = 0
    #     raise Exception('Dataset {} is not supported'.format(args.dataset))

    # if len(train_sets) != len(args.dataset) and len(train_sets) != args.num_domains:
    #     raise Exception('Something went wrong. Please check that your dataset names are valid and your num_domains are set correct')


    # Define new train data set that has all the train sets
    # Define new val data set that has all the val sets
    """ 不用它这个定义的multi domains
    if len(args.dataset) != 1:
        if args.image_uniform_sampling:
            train_set = ConcatDataset(train_sets)
        else:
            train_set = multi_loader.DomainUniformConcatDataset(args, train_sets)
    """

    val_loaders = {}
    for i, val_set in enumerate(val_sets):
        if args.syncbn:
            val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)
        else:
            val_sampler = None
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                                num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                                sampler = val_sampler)
        val_loaders[val_dataset_names[i]] = val_loader

    if args.syncbn:
        train_sampler = DistributedSampler(train_set, pad=True, permutation=True, consecutive_sample=False) # 多卡训练所使用的的采样器
    else:
        train_sampler = None

    # Original —— 总结：① 对于多个训练集，用一个DataLoader装载（先合并）；② 对于多个验证集，用一个DataLoader装载一个验证集，并用一个字典存储多个DataLoaders
    # train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
    #                           num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)

    # Customed —— 总结：① 对于多个训练集，用一个DataLoader装载一个训练集，并用一个list存储多个DataLoader；② 对于多个验证集，用一个DataLoader装载一个验证集，并用一个字典存储多个DataLoaders
    # train_loaders = [DataLoader(train_set, batch_size=args.train_batch_size,
    #                           num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    #                           for train_set in train_sets]

    # print(len(train_loaders[0]))

    # 额外的验证集
    extra_val_loader = {}
    for val_dataset in args.val_dataset:
        extra_val_loader[val_dataset] = create_extra_val_loader(args, val_dataset, val_input_transform, target_transform, val_sampler)

    # 这个不懂是啥子验证集
    covstat_val_loader = {}
    for val_dataset in args.covstat_val_dataset:
        covstat_val_loader[val_dataset] = create_covstat_val_loader(args, val_dataset, val_input_transform, target_transform, val_sampler)

    # return train_loader, val_loaders, train_set, extra_val_loader, covstat_val_loader
    return train_loaders, val_loaders, train_set, extra_val_loader, covstat_val_loader, train_dataset_names
