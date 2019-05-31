import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
import string
import PIL
from PIL import Image
import time
import zipfile
import shutil
import pdb 
import h5py
import random
import matplotlib.pyplot as plt

class CityscapesData(data.Dataset):
    
    def __init__(self, img_folder, label_folder, anno_txt, hdf5_file_name, input_size, cls_num, img_transform, label_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.input_size = input_size
        self.cls_num = cls_num
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.h5_f = h5py.File(hdf5_file_name, 'r')

        # Convert txt file to dict so that can use index to get filename.
        cnt = 0
        self.idx2name_dict = {}
        self.ids = []
        f = open(anno_txt, 'r')
        lines = f.readlines()
        for line in lines:
            row_data = line.split()
            img_name = row_data[0]
            label_name = row_data[1]
            self.idx2name_dict[cnt] = {}
            self.idx2name_dict[cnt]['img'] = img_name
            self.idx2name_dict[cnt]['label'] = label_name
            self.ids.append(cnt)
            cnt += 1

    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)

        # Set the same random seed for img and label transform
        seed = np.random.randint(2147483647)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB') # W X H

        random.seed(seed)
        processed_img = self.img_transform(img) # 3 X H X W

        np_data = self.h5_f['data/'+label_name.replace('/', '_').replace('bin', 'npy')]

        label_data = []
        num_cls = np_data.shape[2]
        for k in xrange(num_cls):
            if np_data[:,:,num_cls-1-k].sum() > 0: # The order is reversed to be consistent with class name idx in official.
                random.seed(seed) # Before transform, set random seed same as img transform, to keep consistent!
                label_tensor = self.label_transform(torch.from_numpy(np_data[:, :, num_cls-1-k]).unsqueeze(0).float())
            else: # ALL zeros, don't need transform, maybe a bit faster?..
                label_tensor = torch.zeros(1, self.input_size, self.input_size).float()
            label_data.append(label_tensor.squeeze(0).long())
        label_data = torch.stack(label_data).transpose(0,1).transpose(1,2) # N X H X W -> H X W X N
        print("label_data.sum():{0}".format(label_data.sum()))
        print("label_data.max():{0}".format(label_data.max()))     
        return processed_img, label_data
        # processed_img: 3 X 472(H) X 472(W)
        # label tensor: 472(H) X 472(W) X 19

    def __len__(self):
        return len(self.ids)

