import numpy as np
from PIL import Image
import os
import shutil
from scipy.misc import imsave

import torch

def convert_num_to_bitfield(label_data, h, w, root_folder, label_png_name, cls_num=19):
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
    label_png_dir = os.path.join("label_img", label_png_name.replace('/', '_')) # eg: label_img/label_img_test_2008_002687
    print("label_png_dir:{0}".format(label_png_dir))
    if not os.path.exists(os.path.join(root_folder, label_png_dir)):
        os.makedirs(os.path.join(root_folder, label_png_dir))
    for cls_idx in xrange(cls_num):
        imsave(os.path.join(root_folder, label_png_dir, str(cls_idx)+'.png'), all_bit_tensor_list[:,:,cls_idx].numpy())

if __name__ == "__main__":
    f = open("test.txt", 'r')
    lines = f.readlines()
    root_folder = "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch-cityscapes/cityscapes-preprocess/data_proc"
    cnt = 0

    for ori_line in lines:
        cnt += 1
        line = ori_line.split()
        bin_name = line[1]
        img_name = line[0]
        
        label_path = os.path.join(root_folder, bin_name) 
        img_path = os.path.join(root_folder, img_name)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size # Notice: not h, w! This is very important! Otherwise, the label is wrong for each pixel.

        label_data = np.fromfile(label_path, dtype=np.uint32)
        label_png_name = img_name.replace('image', 'label_img')
        convert_num_to_bitfield(label_data, h, w, root_folder, label_png_name)
        if cnt % 20 == 0:
            print("{0} have been finished.".format(cnt))

