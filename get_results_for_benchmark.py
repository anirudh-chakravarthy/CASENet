import os
import sys
import argparse

import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import shutil
import h5py
from imageio import imwrite

import torch
from torch import sigmoid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet import CASENet_resnet101
from prep_dataset.prep_cityscapes_dataset import RGB2BGR, ToTorchFormatTensor

import utils.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-m', '--model', type=str,
                        help="path to the pytorch(.pth) containing the trained weights")
    parser.add_argument('-l', '--image_list', type=str, default='',
                        help="list of image files to be tested")
    parser.add_argument('-f', '--image_file', type=str, default='',
                        help="a single image file to be tested")
    parser.add_argument('-d', '--image_dir', type=str, default='',
                        help="root folder of the image files in the list or the single image file")
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help="folder to store the test results")
    args = parser.parse_args(sys.argv[1:])
    
    # load input path
    if os.path.exists(args.image_list):
        with open(args.image_list) as f:
            ori_test_lst = [x.strip().split()[0] for x in f.readlines()]
            if args.image_dir!='':
                test_lst = [
                    args.image_dir+x if os.path.isabs(x)
                    else os.path.join(args.image_dir, x)
                    for x in ori_test_lst]
    else:
        image_file = os.path.join(args.image_dir, args.image_file)
        if os.path.exists(image_file):
            ori_test_list = [args.image_file]
            test_lst = [image_file]
        else:
            raise IOError('nothing to be tested!')
    
    # load net
    num_cls = 19
    model = CASENet_resnet101(pretrained=False, num_classes=num_cls)
    # model = model.cuda()
    model = model.eval()
    # cudnn.benchmark = True
    utils.load_pretrained_model(model, args.model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        for cls_idx in xrange(num_cls):
            dir_path = os.path.join(args.output_dir, str(cls_idx))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    # Define normalization for data    
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])
    crop_size = 632
    
    img_transform = transforms.Compose([
                    RGB2BGR(roll=True),
                    ToTorchFormatTensor(div=False),
                    normalize,
                    ])
    
    for idx_img in xrange(len(test_lst)):
        img = Image.open(test_lst[idx_img]).convert('RGB')
        processed_img = img_transform(img).unsqueeze(0) # 1 X 3 X H X W
        height = processed_img.size()[2]
        width = processed_img.size()[3]
        if crop_size < height or crop_size < width:
            raise ValueError("Input image size must be smaller than crop size!")
        pad_h = crop_size - height
        pad_w = crop_size - width
        padded_processed_img = F.pad(processed_img, (0, pad_w, 0, pad_h), "constant", 0).data
        processed_img_var = utils.check_gpu(None, padded_processed_img) # change None to GPU Id if needed
        score_feats5, score_fuse_feats = model(processed_img_var) # 1 X 19 X CROP_SIZE X CROP_SIZE
        
        score_output = sigmoid(score_fuse_feats.transpose(1,3).transpose(1,2)).squeeze(0)[:height, :width, :] # H X W X 19
        for cls_idx in xrange(num_cls):
            # Convert binary prediction to uint8
            im_arr = np.empty((height, width), np.uint8)
            im_arr = (score_output[:,:,cls_idx].data.cpu().numpy())*255.0
            # print(im_arr)
             
            # Store value into img
            img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
            if not os.path.exists(os.path.join(args.output_dir, str(cls_idx))):
                os.makedirs(os.path.join(args.output_dir, str(cls_idx)))
            imwrite(os.path.join(args.output_dir, str(cls_idx), img_base_name_noext+'.png'), im_arr)
            print 'processed: '+test_lst[idx_img]
    
    print 'Done!'

