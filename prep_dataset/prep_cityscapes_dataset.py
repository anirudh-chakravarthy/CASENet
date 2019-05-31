import os
import numpy as np
import time
import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
sys.path.append("../")

from dataloader.cityscapes_data import CityscapesData

import config

class RGB2BGR(object):
    """
    Since we use pretrained model from Caffe, need to be consistent with Caffe model.
    Transform RGB to BGR.
    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        if img.mode == 'L':
            return np.concatenate([np.expand_dims(img, 2)], axis=2) 
        elif img.mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(img)[:, :, ::-1]], axis=2)
            else:
                return np.concatenate([np.array(img)], axis=2)

class ToTorchFormatTensor(object):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] or [0, 255]. 
    """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        return img.float().div(255) if self.div else img.float()

def get_dataloader(args):
    # Define data files path.
    root_img_folder = "/ais/gobi4/fashion/edge_detection/data_aug" 
    root_label_folder = "/ais/gobi4/fashion/edge_detection/data_aug"
    train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_train_aug.txt"
    val_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"
    train_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/train_aug_label_binary_np.h5"
    val_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/test_label_binary_np.h5"

    input_size = 472
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

    train_augmentation = transforms.Compose([transforms.RandomResizedCrop(input_size, scale=(0.75,1.0), ratio=(0.75,1.0)), transforms.RandomHorizontalFlip()])
    train_label_augmentation = transforms.Compose([transforms.RandomResizedCrop(input_size, scale=(0.75,1.0), ratio=(0.75,1.0), interpolation=PIL.Image.NEAREST), \
                                transforms.RandomHorizontalFlip()])

    train_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        train_hdf5_file,
        input_size,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        train_augmentation,
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        train_label_augmentation,
                        transforms.ToTensor(),
                        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        val_hdf5_file,
        input_size,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
                        transforms.ToTensor(),
                        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size/2, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    args = config.get_args()
    args.batch_size = 1
    train_loader, val_loader = get_dataloader(args)
    for i, (img, target) in enumerate(val_loader):
        print("target.size():{0}".format(target.size()))
        print("target:{0}".format(target))
        break;
