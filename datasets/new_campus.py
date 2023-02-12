"""
campusE1 Dataset Loader
"""
import logging
import json
import os
import numpy as np
from PIL import Image, ImageCms
from skimage import color

from torch.utils import data
import torch
import torchvision.transforms as transforms
import datasets.uniform as uniform
import datasets.cityscapes_labels as cityscapes_labels
import datasets.new_campus_labels as campus_labels
import scipy.misc as m

from config import cfg

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
trainid_to_trainid = cityscapes_labels.trainId2trainId
color_to_trainid = campus_labels.color2trainId
num_classes = 31
ignore_label = 255
# root = cfg.DATASET.campusE1_DIR
root = cfg.DATASET.New_CampusE1_DIR
# root = "/data/user21100736/datasets/New_CampusE1/E1_1F"
img_postfix = '.png'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def add_items(items, aug_items, videos, img_path, mask_path, mask_postfix, mode, maxSkip, env):
    """

    Add More items ot the list from the augmented dataset
    """
    if mode == "val":
        if env == "E1G":
            videos = ['val/E1G']
        elif env == "E1_1F":
            videos = ['val/E1_1F']
        elif env == "E1_2F":
            videos = ['val/E1_2F']
        # else: "videos" remains the same.

    for v in videos:
        v_items = [name.split(img_postfix)[0] for name in
                   os.listdir(os.path.join(img_path, v))]
        for it in v_items:
            item = (os.path.join(img_path, v, it + img_postfix),
                    os.path.join(mask_path, v, it + mask_postfix))
            ########################################################
            ###### dataset augmentation ############################
            ########################################################
            # if mode == "train" and maxSkip > 0:
            #     new_img_path = os.path.join(aug_root, 'leftImg8bit_trainvaltest', 'leftImg8bit')
            #     new_mask_path = os.path.join(aug_root, 'gtFine_trainvaltest', 'gtFine')
            #     file_info = it.split("_")
            #     cur_seq_id = file_info[-1]

            #     prev_seq_id = "%06d" % (int(cur_seq_id) - maxSkip)
            #     next_seq_id = "%06d" % (int(cur_seq_id) + maxSkip)
            #     prev_it = file_info[0] + "_" + file_info[1] + "_" + prev_seq_id
            #     next_it = file_info[0] + "_" + file_info[1] + "_" + next_seq_id
            #     prev_item = (os.path.join(new_img_path, c, prev_it + img_postfix),
            #                  os.path.join(new_mask_path, c, prev_it + mask_postfix))
            #     if os.path.isfile(prev_item[0]) and os.path.isfile(prev_item[1]):
            #         aug_items.append(prev_item)
            #     next_item = (os.path.join(new_img_path, c, next_it + img_postfix),
            #                  os.path.join(new_mask_path, c, next_it + mask_postfix))
            #     if os.path.isfile(next_item[0]) and os.path.isfile(next_item[1]):ss
            #         aug_items.append(next_item)
            items.append(item)
    # items.extend(extra_items)


def make_cv_splits(img_dir_name, split=None):
    """
    Create splits of train/val data.
    A split is a lists of videos.
    """
    trn_path = os.path.join(root, img_dir_name, 'train')
    val_path = os.path.join(root, img_dir_name, 'val')

    trn_videos = ['train/' + v for v in os.listdir(trn_path)]
    val_videos = ['val/' + v for v in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_videos = sorted(trn_videos)

    all_videos = val_videos + trn_videos
    num_val_videos = len(val_videos)
    num_videos = len(all_videos)

    cv_splits = []
    for split_idx in range(cfg.DATASET.CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_videos // cfg.DATASET.CV_SPLITS
        for j in range(num_videos):
            if j >= offset and j < (offset + num_val_videos):
                split['val'].append(all_videos[j])
            else:
                split['train'].append(all_videos[j])
        cv_splits.append(split)

    return cv_splits


def make_split_coarse(img_path):
    """
    Create a train/val split for coarse
    return: city split in train
    """
    all_videos = os.listdir(img_path)
    all_videos = sorted(all_videos)  # needs to always be the same
    val_videos = []  # Can manually set videos to not be included into train split

    split = {}
    split['val'] = val_videos
    split['train'] = [v for v in all_videos if v not in val_videos]
    return split


def make_test_split(img_dir_name, split=None):
    test_path = os.path.join(root, split, img_dir_name, 'test')
    test_cities = ['test/' + c for c in os.listdir(test_path)]

    return test_cities


def make_dataset(mode, weather, maxSkip=0, fine_coarse_mult=6, cv_split=0, split=None, env=None):
    """
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    rgb_anon_trainextra/rgb_anon/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    aug_items = []

    assert (cv_split == 0)
    assert mode in ['train', 'val']
    img_path = os.path.join(root, "images") # 图片所处地址（该目录下有train val test；接下来是各个videos；然后才是图片）
    mask_path = os.path.join(root, "labels_color") # 标注所处地址（该目录下有train val；接下来是各个videos；然后才是标注）
    mask_postfix = '_TrainIds.png' # 标注的后缀
    cv_splits = make_cv_splits("images", split=split) # 根据图片所处路径划分train和val的videos
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        if mode == 'test':
            cv_splits = make_test_split("images")
            add_items(items, aug_items, cv_splits, img_path, mask_path,
                      mask_postfix, mode, maxSkip)
        else:
            logging.info('{} videos: '.format(mode) + str(cv_splits[cv_split][mode]))
            add_items(items, aug_items, cv_splits[cv_split][mode], img_path, mask_path,
                      mask_postfix, mode, maxSkip, env) # 根据mode和基于前面划分好的train和val，追踪到叶路径（最终分别存放图片和标注的地方），然后形成“一个图片 + 一个标注”的每个item

    # logging.info('ACDC-{}: {} images'.format(mode, len(items)))
    logging.info('CampusE1-{}: {} images'.format(mode, len(items) + len(aug_items)))
    return items, aug_items

class CampusE1Uniform(data.Dataset):
    """
    Please do not use this for AGG
    """

    def __init__(self, mode, maxSkip=0, joint_transform_list=None, sliding_crop=None,
                 transform=None, target_transform=None, target_aux_transform=None, dump_images=False,
                 cv_split=None, class_uniform_pct=0.5, class_uniform_tile=1024,
                 test=False, coarse_boost_classes=None, is_additional=False, image_in=False,
                 extract_feature=False, split=None, env=None):
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.target_aux_transform = target_aux_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.coarse_boost_classes = coarse_boost_classes
        self.is_additional = is_additional
        self.image_in = image_in
        self.extract_feature = extract_feature
        self.split = split
        self.env = env

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS)
        else:
            self.cv_split = 0

        self.imgs, self.aug_imgs = make_dataset(mode, self.maxSkip, cv_split=self.cv_split, split=self.split, env=self.env)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for fine data
        json_fn = 'New_CampusE1_new_{}_cv{}_tile{}.json'.format(
            self.mode, self.cv_split, self.class_uniform_tile)
        if os.path.isfile(json_fn):
            with open(json_fn, 'r') as json_data:
                centroids = json.load(json_data)
            for idx in centroids:
                print("###### centroids", idx)

            self.centroids = {int(idx): centroids[idx] for idx in centroids}
        else:
            self.centroids = uniform.class_centroids_all_from_color(
                self.imgs,
                num_classes,
                id2trainid=color_to_trainid,
                tile_size=class_uniform_tile)
            with open(json_fn, 'w') as outfile:
                json.dump(self.centroids, outfile, indent=4)

        self.fine_centroids = self.centroids.copy()

        self.build_epoch() # 会根据num_classes和centroids的结果更改imgs_uniform的数量，也即一个epoch训练样本的数量。

    def cities_uniform(self, imgs, name):
        """ list out cities in imgs_uniform """
        cities = {}
        for item in imgs:
            img_fn = item[0]
            img_fn = os.path.basename(img_fn)
            city = img_fn.split('_')[0]
            cities[city] = 1
        city_names = cities.keys()
        logging.info('Cities for {} '.format(name) + str(sorted(city_names)))

    def build_epoch(self, cut=False):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        if self.class_uniform_pct > 0:
            if self.is_additional:
                if cut:
                    # after max_cu_epoch, we only fine images to fine tune
                    self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                            self.fine_centroids,
                                                            num_classes,
                                                            cfg.CLASS_UNIFORM_PCT_ADD)
                else:
                    self.imgs_uniform = uniform.build_epoch(self.imgs + self.aug_imgs,
                                                            self.centroids,
                                                            num_classes,
                                                            cfg.CLASS_UNIFORM_PCT_ADD)
            else:
                if cut:
                    # after max_cu_epoch, we only fine images to fine tune
                    self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                            self.fine_centroids,
                                                            num_classes,
                                                            cfg.CLASS_UNIFORM_PCT)
                else:
                    self.imgs_uniform = uniform.build_epoch(self.imgs + self.aug_imgs,
                                                            self.centroids,
                                                            num_classes,
                                                            cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index): # 不是直接改变数据集样本结构的，只是在读取数据时使用
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem
        img, mask = Image.open(img_path).convert('RGB'), m.imread(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # print(img.size, mask[:,:,0].shape)
        while (img.size[1], img.size[0]) != mask[:,:,0].shape:
            print("Error!!", img.size, mask[:,:,0].shape, img_name)
            print("Dropping ", str(index))
            # index = index + 1
            if index + 1 == len(self.imgs):
                index = 0
            else:
                index += 1
            img_path, mask_path = self.imgs[index]
            img, mask = Image.open(img_path).convert('RGB'), m.imread(mask_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]

        image_size = mask[:,:,0].shape
        mask_copy = np.full(image_size, ignore_label, dtype=np.uint8)

        for k, v in color_to_trainid.items():
            if v != 255 and v != -1:
                mask_copy[(mask == np.array(k))[:,:,0] & (mask == np.array(k))[:,:,1] & (mask == np.array(k))[:,:,2]] = v

        # for k, v in color_to_trainid.items():
        #     mask_copy[(mask == np.array(k))[:,:,0]] = v

        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # mask = np.array(mask)
        # mask_copy = mask.copy()
        # for k, v in trainid_to_trainid.items():
        #     mask_copy[mask == k] = v
        # mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.extract_feature is not True:
            if self.joint_transform_list is not None:
                for idx, xform in enumerate(self.joint_transform_list):
                    if idx == 0 and centroid is not None:
                        # HACK
                        # We assume that the first transform is capable of taking
                        # in a centroid
                        img, mask = xform(img, mask, centroid)
                    else:
                        img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)

        rgb_mean_std_gt = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_gt = transforms.Normalize(*rgb_mean_std_gt)(img)

        rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.image_in:
            eps = 1e-5
            rgb_mean_std = ([torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])],
                    [torch.std(img[0])+eps, torch.std(img[1])+eps, torch.std(img[2])+eps])
        img = transforms.Normalize(*rgb_mean_std)(img)

        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, img_name, mask_aux

    def __len__(self):
        return len(self.imgs_uniform)

class CampusE1(data.Dataset):

    def __init__(self, mode, maxSkip=0, joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None, target_aux_transform=None, dump_images=False,
                 cv_split=None, eval_mode=False,
                 eval_scales=None, eval_flip=False, image_in=False, extract_feature=False, split=None, env=None):
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.target_aux_transform = target_aux_transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        self.image_in = image_in
        self.extract_feature = extract_feature
        self.split = split
        self.env = env

        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS)
        else:
            self.cv_split = 0

        self.imgs, _ = make_dataset(mode, self.maxSkip, cv_split=self.cv_split, split=self.split, env=self.env)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, mask, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool) + 1):
            imgs = []
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w, h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img = img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(final_tensor)
            return_imgs.append(imgs)
        return return_imgs, mask

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), m.imread(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        while (img.size[1], img.size[0]) != mask[:,:,0].shape:
            print("Error!!", img.size, mask[:,:,0].shape, img_name)
            print("Dropping ", str(index))
            # index = index + 1
            if index + 1 == len(self.imgs):
                index = 0
            else:
                index += 1
            img_path, mask_path = self.imgs[index]
            img, mask = Image.open(img_path).convert('RGB'), m.imread(mask_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]

        image_size = mask[:,:,0].shape
        mask_copy = np.full(image_size, ignore_label, dtype=np.uint8)

        for k, v in color_to_trainid.items():
            if v != 255 and v != -1:
                mask_copy[(mask == np.array(k))[:,:,0] & (mask == np.array(k))[:,:,1] & (mask == np.array(k))[:,:,2]] = v

        # for k, v in color_to_trainid.items():
        #     mask_copy[(mask == np.array(k))[:,:,0]] = v

        if self.eval_mode:
            return [transforms.ToTensor()(img)], self._eval_get_item(img, mask_copy,
                                                                     self.eval_scales,
                                                                     self.eval_flip), img_name

        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.extract_feature is not True:
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        rgb_mean_std_gt = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_gt = transforms.Normalize(*rgb_mean_std_gt)(img)

        rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.image_in:
            eps = 1e-5
            rgb_mean_std = ([torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])],
                    [torch.std(img[0])+eps, torch.std(img[1])+eps, torch.std(img[2])+eps])
        img = transforms.Normalize(*rgb_mean_std)(img)

        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        return img, mask, img_name, mask_aux

    def __len__(self):
        return len(self.imgs)