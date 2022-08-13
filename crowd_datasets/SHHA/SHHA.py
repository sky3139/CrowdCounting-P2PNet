import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "shanghai_tech_part_a_train.list"
        self.eval_list = "shanghai_tech_part_a_test.list"
        # there may exist multiple list files
        # print(data_root)
        if train:
            npy_path=data_root+"/train/*.npy"
        else:
            npy_path=data_root+"/train/*.npy"
        self.gt_list =sorted( glob.glob(npy_path))  # change to npy for gt_list

        self.img_map = {}
        self.img_gt_list = []
        # loads the image/gt pairs
        for i, train_list in enumerate(self.gt_list):
            if train and i%2==0:
                rgb_path = train_list.replace('GT', 'RGB').replace('npy', 'jpg')
                t_path = train_list.replace('GT', 'T').replace('npy', 'jpg')
                self.img_gt_list.append([rgb_path,train_list])
            if train==False and i%2==1:
                rgb_path = train_list.replace('GT', 'RGB').replace('npy', 'jpg')
                t_path = train_list.replace('GT', 'T').replace('npy', 'jpg')
                self.img_gt_list.append([rgb_path,train_list])
            # print(rgb_path,train_list)

            # self.img_map[rgb_path] =train_list
        # number of samples
        self.nSamples = len(self.img_gt_list)
        # print(self.gt_list)
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_gt_list[index][0]
        gt_path = self.img_gt_list[index][1]
        # print(img_path,gt_path)
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        # print(point)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)
        # print(point)
        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]
        
        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('_RGB.')[0])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # print(gt_path)
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points  = np.load(gt_path)
    # print(gt_path,points)
    return img, np.array(points).astype("float64")

# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den