import numpy as np
from PIL import Image
import os
import zipfile
import shutil
import h5py
from tqdm import tqdm

import torch

def convert_num_to_bitfield(label_data, h, w, npz_name, root_folder, h5_file, cls_num=19):
    label_list = list(label_data)
    all_bit_tensor_list = []
    for n in label_list: # Iterate in each pixel
        # Convert a value to binary format in each bit.
        bitfield = np.asarray([int(digit) for digit in bin(n)[2:]])
        bit_tensor = torch.from_numpy(bitfield)
        actual_len = bit_tensor.size()[0]
        padded_bit_tensor = torch.cat((torch.zeros(cls_num-actual_len).byte(), bit_tensor.byte()), dim=0)
        all_bit_tensor_list.append(padded_bit_tensor)
    all_bit_tensor_list = torch.stack(all_bit_tensor_list).view(h, w, cls_num)
    h5_file.create_dataset('data/'+npz_name.replace('/', '_'), data=all_bit_tensor_list.numpy())

if __name__ == "__main__":
    f = open("/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch-cityscapes/cityscapes-preprocess/data_proc/val.txt", 'r')
    lines = f.readlines()
    root_folder = "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch-cityscapes/cityscapes-preprocess/data_proc/"

    h5_file = h5py.File("val_label_binary_np.h5", 'w')
    for ori_line in tqdm(lines):
        line = ori_line.split()
        bin_name = line[1]
        img_name = line[0]
        
        label_path = os.path.join(root_folder, bin_name) 
        img_path = os.path.join(root_folder, img_name)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size # Notice: not h, w! This is very important! Otherwise, the label is wrong for each pixel.

        label_data = np.fromfile(label_path, dtype=np.uint32)
        npz_name = bin_name.replace("bin", "npy")
        convert_num_to_bitfield(label_data, h, w, npz_name, root_folder, h5_file)
