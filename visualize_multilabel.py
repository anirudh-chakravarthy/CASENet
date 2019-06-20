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

import torch
from torch import sigmoid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet import CASENet_resnet101
from prep_dataset.prep_cityscapes_dataset import RGB2BGR, ToTorchFormatTensor

import utils.utils as utils

def get_cityscapes_class_names():
    return ['road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle']

def normalized_feature_map(fmap):
    fmap_min = fmap.min()
    fmap_max = fmap.max()
    fmap = (fmap-fmap_min)/(fmap_max-fmap_min)
    return fmap

# color map for each trainId, from the official cityscapes script
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
def get_colors():
	color_dict = {}
	color_dict[0] = [128, 64, 128]
	color_dict[1] = [244, 35, 232]
	color_dict[2] = [70, 70, 70]
	color_dict[3] = [102, 102, 156]
	color_dict[4] = [190, 153, 153]
	color_dict[5] = [153, 153, 153]
	color_dict[6] = [250, 170, 30]
	color_dict[7] = [220, 220, 0]
	color_dict[8] = [107, 142, 35]
	color_dict[9] = [152, 251, 152]
	color_dict[10] = [70, 130, 180]
	color_dict[11] = [220, 20, 60]
	color_dict[12] = [255,  0,  0]
	color_dict[13] = [0, 0, 142]
	color_dict[14] = [0, 0, 70]
	color_dict[15] = [0, 60, 100]
	color_dict[16] = [0, 80, 100]
	color_dict[17] = [0, 0, 230]
	color_dict[18] = [119, 11, 32]
	return color_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-m', '--model', type=str,
                        help="path to the Pytorch model(.pth) containing the trained weights")
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
            ori_test_list = [x.strip().split()[0] for x in f.readlines()]
            if args.image_dir!='':
                test_list = [
                    args.image_dir+x if os.path.isabs(x)
                    else os.path.join(args.image_dir, x)
                    for x in ori_test_list]
    else:
        image_file = os.path.join(args.image_dir, args.image_file)
        if os.path.exists(image_file):
            ori_test_list = [args.image_file]
            test_list = [image_file]
        else:
            raise IOError('nothing to be tested!')

    # load network
    num_cls = 19
    model = CASENet_resnet101(pretrained=False, num_classes=num_cls)
    # model = model.cuda()
    model = model.eval()
    # cudnn.benchmark = True
    utils.load_pretrained_model(model, args.model)

    cls_names = get_cityscapes_class_names()
    color_dict = get_colors()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

     # Define normalization for data 
    input_size = 472
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])
    
    img_transform = transforms.Compose([
                    transforms.Resize([input_size, input_size]),
                    RGB2BGR(roll=True),
                    ToTorchFormatTensor(div=False),
                    normalize,
                    ])
    label_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
                    transforms.ToTensor(),
                    ])

    h5_f = h5py.File("./utils/val_label_binary_np.h5", 'r')
    
    for idx_img in range(len(test_list)):
        img = Image.open(test_list[idx_img]).convert('RGB')
        processed_img = img_transform(img).unsqueeze(0)
        processed_img = utils.check_gpu(None, processed_img)    
        score_feats1, score_feats2, score_feats3, score_feats5, score_fuse_feats = model(processed_img, for_vis=True)

        # Load numpy from hdf5 for gt.
        np_data = h5_f['data/'+ori_test_list[idx_img].replace('leftImg8bit', 'gtFine').replace('/', '_').replace('.png', '_edge.npy')]
        label_data = []
        num_cls = np_data.shape[2]
        for k in range(num_cls):
            if np_data[:,:,num_cls-1-k].sum() > 0:
                label_tensor = label_transform(torch.from_numpy(np_data[:, :, num_cls-1-k]).unsqueeze(0).float())
            else: # ALL zeros, don't need transform, maybe a bit faster?..
                label_tensor = torch.zeros(1, input_size, input_size).float()
            label_data.append(label_tensor.squeeze(0).long())
        label_data = torch.stack(label_data).transpose(0,1).transpose(1,2) # N X H X W -> H X W X N
    
        img_base_name_noext = os.path.splitext(os.path.basename(test_list[idx_img]))[0]
        img_base_name_noext = img_base_name_noext.replace('_leftImg8bit', '') 
       
        score_feats_list = [score_feats1, score_feats2, score_feats3]
        score_feats_str_list = ['feats1', 'feats2', 'feats3'] 

		# visualize side edge activation
        for i in range(len(score_feats_list)):
            feature = score_feats_list[i]
            feature_str = score_feats_str_list[i]
    
            side = normalized_feature_map(feature.data[0][0, :, :].cpu().numpy())
            im = (side*255).astype(np.uint8)
            if not os.path.exists(os.path.join(args.output_dir, img_base_name_noext)):
                os.makedirs(os.path.join(args.output_dir, img_base_name_noext))
            cv2.imwrite(
                os.path.join(args.output_dir, img_base_name_noext, img_base_name_noext+'_'+feature_str+'.png'),
                im)

        # visualize side class activation
        side_cls = normalized_feature_map(np.transpose(score_feats5.data[0].cpu().numpy(), (1, 2, 0)))
        for idx_cls in range(num_cls):
            side_cls_i = side_cls[:, :, idx_cls]
            im = (side_cls_i * 255).astype(np.uint8)
            if not os.path.exists(os.path.join(args.output_dir, img_base_name_noext)):
                os.makedirs(os.path.join(args.output_dir, img_base_name_noext))
            cv2.imwrite(
                os.path.join(args.output_dir, img_base_name_noext, img_base_name_noext+'_'+'feats5'+'_'+cls_names[num_cls-idx_cls-1]+'.png'),
                im)
    
        # visualize predicted class and contours
        score_output = sigmoid(score_fuse_feats.transpose(1,3).transpose(1,2)).data[0].cpu().numpy()
        for idx_cls in range(num_cls):
            r = np.zeros((score_output.shape[0], score_output.shape[1]))
            g = np.zeros((score_output.shape[0], score_output.shape[1]))
            b = np.zeros((score_output.shape[0], score_output.shape[1]))
            rgb = np.zeros((score_output.shape[0], score_output.shape[1], 3))
            score_pred = score_output[:, :, idx_cls]
            score_pred_flag = (score_pred>0.5)
            r[score_pred_flag==1] = color_dict[idx_cls][0]
            g[score_pred_flag==1] = color_dict[idx_cls][1]
            b[score_pred_flag==1] = color_dict[idx_cls][2]
            r[score_pred_flag==0] = 255
            g[score_pred_flag==0] = 255
            b[score_pred_flag==0] = 255
            rgb[:,:,0] = (r/255.0)
            rgb[:,:,1] = (g/255.0)
            rgb[:,:,2] = (b/255.0)
            if not os.path.exists(os.path.join(args.output_dir, img_base_name_noext)):
                os.makedirs(os.path.join(args.output_dir, img_base_name_noext))
            plt.imsave(os.path.join(args.output_dir, img_base_name_noext, img_base_name_noext+'_fused_pred_'+cls_names[idx_cls]+'.png'), rgb) 

            gray = np.zeros(r.shape, dtype=np.uint8)
            gray[score_pred_flag==0] = 0
            gray[score_pred_flag==1] = 127

            # create border for image
            top = 1
            bottom = top
            left = 1
            right = left
            gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (127,))

            # morphological operator
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            # dilated = cv2.dilate(gray, kernel)
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) # Closing Morphological Operator
            # opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel) # Opening Morphological Operator
            _, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # select contour based on area and arc length
            # print(len(cnts)) 
            # cnts = [c for c in cnts if (cv2.arcLength(c, True) < 1880.0 and cv2.arcLength(c, True) >= 10.0)]
            cnts = [c for c in cnts if (cv2.arcLength(c, True) < 1880.0 and cv2.contourArea(c, True) >= 150.0)]
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
            # print(cls_names[idx_cls])
            # for c in cnts:
            #     print(cv2.arcLength(c, True), cv2.contourArea(c))   
            # cv2.drawContours(gray, cnts, -1, (255,255,0), 3)
            
            resizedimg = img.resize((472,472), resample=PIL.Image.BILINEAR)
            opencv_image = np.array(resizedimg) 
            opencv_image = opencv_image[:, :, ::-1].copy() 
            cv2.drawContours(opencv_image, cnts, -1, (255,255,0), 1)
            opencv_image = cv2.resize(opencv_image, img.size, interpolation = cv2.INTER_CUBIC)
            
            if not os.path.exists(os.path.join(args.output_dir, img_base_name_noext)):
                os.makedirs(os.path.join(args.output_dir, img_base_name_noext))
            cv2.imwrite(os.path.join(args.output_dir, img_base_name_noext, img_base_name_noext+'_'+cls_names[idx_cls]+'.png'), opencv_image)

        # gt visualization
        gt_data = label_data.numpy() 
        for idx_cls in range(num_cls):
            r = np.zeros((gt_data.shape[0], gt_data.shape[1]))
            g = np.zeros((gt_data.shape[0], gt_data.shape[1]))
            b = np.zeros((gt_data.shape[0], gt_data.shape[1]))
            rgb = np.zeros((gt_data.shape[0], gt_data.shape[1], 3))
            score_pred_flag = gt_data[:, :, idx_cls]
            r[score_pred_flag==1] = color_dict[idx_cls][0]
            g[score_pred_flag==1] = color_dict[idx_cls][1]
            b[score_pred_flag==1] = color_dict[idx_cls][2]
            r[score_pred_flag==0] = 255
            g[score_pred_flag==0] = 255
            b[score_pred_flag==0] = 255
            rgb[:,:,0] = (r/255.0)
            rgb[:,:,1] = (g/255.0)
            rgb[:,:,2] = (b/255.0)
            plt.imsave(os.path.join(args.output_dir, img_base_name_noext, img_base_name_noext+'_gt_'+cls_names[idx_cls]+'.png'), rgb) 
    
        print('processed: '+test_list[idx_img])
    
    print('Done!')
