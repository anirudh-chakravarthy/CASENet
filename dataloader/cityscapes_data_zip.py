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

class CityscapesData(data.Dataset):
    
    def __init__(self, img_folder, label_folder, anno_txt, cls_num, img_transform, label_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.cls_num = cls_num
        self.img_transform = img_transform
        self.label_transform = label_transform

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
        index = 0
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, label_name)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB') # W X H
        processed_img = self.img_transform(img) # 3 X H X W

        # Load label into tensor
        # Read zip file and extract to npy, then load npy to numpy, delete numpy finally.
        zip_file = zipfile.ZipFile(label_path, 'r')
        tmp_folder = os.path.join("/ais/gobi4/fashion/edge_detection/tmp_npy", label_name.split('/')[-2], label_name.split('/')[-1])
        extract_data = zip_file.extract("label", tmp_folder)
        np_data = np.load(os.path.join(tmp_folder, "label")) # H X W X NUM_CLASSES
        
        label_data = []
        for k in xrange(np_data.shape[2]):
            label_tensor = self.label_transform(torch.from_numpy(np_data[:, :, k]).unsqueeze(0).float())
            label_data.append(label_tensor.squeeze(0).long())
        label_data = torch.stack(label_data).transpose(0,1).transpose(1,2) # N X H X W -> H X W X N
        shutil.rmtree(tmp_folder) 
        return processed_img, label_data
        # processed_img: 3 X 472(H) X 472(W)
        # label tensor: 472(H) X 472(W) X 19

    def __len__(self):
        return len(self.ids)

