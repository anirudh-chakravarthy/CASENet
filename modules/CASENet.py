import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import torchvision.models as models
import sys
sys.path.append("../")

import numpy as np
import utils.utils as utils
import os

def gen_mapping_layer_name(model):
    """
    Generate mapping for name in pytorch with name in Caffe to load numpy file we transformed.
    Notice that BatchNorm in pytorch is very different from Caffe. 
    bn.weight --> scale_layer.weight (Caffe)
    bn.bias --> scale_layer.bias (Caffe)
    bn.running_mean --> bn_layer, blob[0]/blob[2]
    bn.running_var --> bn_layer, blob[1]/blob[2]
    """
    layer_to_name_dict = {} # key is module name in currrent model, value is the filename of numpy
    for (m_name, m) in model.named_parameters():
        if not "res" in m_name: # This case, name is totally the same.
            if "bn" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.replace("bn", "scale").replace(".weight", "")+"_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.replace("bn", "scale").replace(".bias", "")+"_1"
            else:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.replace(".weight", "")+"_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.replace(".bias", "")+"_1"
        else:
            if "res2" in m_name or "res5" in m_name:
                anno_dict = {'0':"a", '1':"b", '2':"c"}
            elif "res3" in m_name or "res4" in m_name:
                anno_dict = {}
                for k in xrange(23):
                    if k == 0:
                        anno_dict[str(k)] = "a"
                    else:
                        anno_dict[str(k)] = "b"+str(k)
                         
            label_idx = m_name.split('.')[1] # Represnet 0 in res2.0
            label_anno = anno_dict[label_idx] # Represent "a", "b", "c" or "b1", etc 
            
            if "downsample.0" in m_name: # For the convolution in another branch
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch1_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch1_1"

            if "downsample.1" in m_name: # For the BN layer in another branch
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch1_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch1_1"

            if "conv1" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch2a_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch2a_1"
            if "conv2" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch2b_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch2b_1"
            if "conv3" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch2c_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0]+label_anno+"_branch2c_1"

            # For BN layer
            if "bn1" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch2a_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch2a_1"
            if "bn2" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch2b_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch2b_1"
            if "bn3" in m_name:
                if "weight" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch2c_0"
                if "bias" in m_name:
                    layer_to_name_dict[m_name] = m_name.split('.')[0].replace("res", "scale")+label_anno+"_branch2c_1"
            
    # For BN running_mean/var, since it's not in parameters, we need to deal with it here.
    for k in layer_to_name_dict.keys():
        if ("bn" in k or "downsample.1" in k) and ("weight" in k):
            avg_key_name = k.replace(k.split('.')[-1], "running_mean")
            avg_name = [layer_to_name_dict[k][:-1].replace("scale", "bn")+"0", layer_to_name_dict[k][:-1].replace("scale", "bn")+"2"]
            var_key_name = k.replace(k.split('.')[-1], "running_var")
            var_name = [layer_to_name_dict[k][:-1].replace("scale", "bn")+"1", layer_to_name_dict[k][:-1].replace("scale", "bn")+"2"]
            layer_to_name_dict[avg_key_name] = avg_name
            layer_to_name_dict[var_key_name] = var_name
    
    print("Total number is:{0}".format(len(layer_to_name_dict.keys())))
    for k in layer_to_name_dict.keys():
        print("k:{0}, v:{1}".format(k, layer_to_name_dict[k]))
    
    return layer_to_name_dict

def load_npy_to_layer(model, layer_to_name_dict, npy_folder, loaded_model_path):
    # Detect if all the npy name in the dict
    pretrained_dict = model.state_dict()
    for (m_name, m_data) in model.state_dict().items():
        if m_name in layer_to_name_dict:
            if not ("running_mean" in m_name or "running_var" in m_name):
                npy_path = os.path.join(npy_folder, layer_to_name_dict[m_name]+".npy")
                np_data = np.load(open(npy_path, 'r'))
                pretrained_dict[m_name] = torch.from_numpy(np_data)
                print("{0} loaded successfully".format(m_name))
            else:
                # Need to calculate the avg, var
                up_path = os.path.join(npy_folder, layer_to_name_dict[m_name][0]+".npy")
                up_data = np.load(open(up_path, 'r'))
                down_path = os.path.join(npy_folder, layer_to_name_dict[m_name][1]+".npy")
                down_data = np.load(open(down_path, 'r'))
                pretrained_dict[m_name] = torch.from_numpy(up_data/down_data)
                print("{0} loaded successfully".format(m_name))

    # Store model into pth file.
    model.load_state_dict(pretrained_dict)
    torch.save({'state_dict': model.state_dict()}, loaded_model_path)

# Bilinear initialization for ConvTranspose2D layer
def init_bilinear(arr):
    weight = np.zeros(np.prod(arr.size()), dtype='float32')
    shape = arr.size()
    f = np.ceil(shape[3] / 2.)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(np.prod(shape)):
        x = i % shape[3]
        y = (i / shape[3]) % shape[2]
        weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    
    return torch.from_numpy(weight.reshape(shape))

def set_require_grad_to_false(m):
    for param in m.parameters():
        param.requires_grad = False

class ScaleLayer(nn.Module):
    """
    This layer is not used. Since BN(pytorch) = BN + Scale (Caffe).
    Before we used BN+Scale in pytorch, this is a bug. :/
    """
    def __init__(self, size, init_value=0.001):
        """
        Adopted from https://discuss.pytorch.org/t/is-scale-layer-available-in-pytorch/7954/6
        """
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]*size))
        self.bias = nn.Parameter(torch.FloatTensor([init_value]*size))

    def forward(self, input_data):
        return input_data * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

class CropLayer(nn.Module):
    
    def __init__(self):
        super(CropLayer, self).__init__()

    def forward(self, input_data, offset):
        """
        Currently, only for specific axis, the same offset. Assume for h, w dim.
        """
        cropped_data = input_data[:, :, offset:-offset, offset:-offset]
        return cropped_data

class SliceLayer(nn.Module):

    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, input_data):
        """
        slice into several single piece in a specific dimension. Here for dim=1
        """
        sliced_list = []
        for idx in xrange(input_data.size()[1]):
            sliced_list.append(input_data[:, idx, :, :].unsqueeze(1))

        return sliced_list

class ConcatLayer(nn.Module):

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, input_data_list, dim):
        concat_feats = torch.cat((input_data_list), dim=dim)
        return concat_feats

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, special_case=False):
        """
        special case only for res5a branch
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        set_require_grad_to_false(self.bn1)

        if special_case:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   dilation=4, padding=4, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=2, padding=2, bias=False)
   
        self.bn2 = nn.BatchNorm2d(planes)
        set_require_grad_to_false(self.bn2)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        set_require_grad_to_false(self.bn3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn_conv1 = nn.BatchNorm2d(64)
        set_require_grad_to_false(self.bn_conv1)
   
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.res2 = self._make_layer(block, 64, layers[0])
        self.res3 = self._make_layer(block, 128, layers[1], stride=2)
        self.res4 = self._make_layer(block, 256, layers[2], stride=2)
        self.res5 = self._make_layer(block, 512, layers[3], stride=1, special_case=True) # Notice official resnet is 2, but CASENet here is 1.

        # Added by CASENet to get feature map from each branch in different scales.
        self.score_edge_side1 = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.score_edge_side2 = nn.Conv2d(256, 1, kernel_size=1, bias=True)
        self.upsample_edge_side2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, bias=False)
        set_require_grad_to_false(self.upsample_edge_side2)
        
        self.score_edge_side3 = nn.Conv2d(512, 1, kernel_size=1, bias=True)
        self.upsample_edge_side3 = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=4, bias=False)
        set_require_grad_to_false(self.upsample_edge_side3) 
        
        self.score_cls_side5 = nn.Conv2d(2048, num_classes, kernel_size=1, bias=True)
        self.upsample_cls_side5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, groups=num_classes, bias=False)
        set_require_grad_to_false(self.upsample_cls_side5) 

        self.ce_fusion = nn.Conv2d(num_classes*4, num_classes, kernel_size=1, groups=num_classes, bias=True)

        # Define crop, concat layer
        self.crop_layer = CropLayer()
        self.slice_layer = SliceLayer()
        self.concat_layer = ConcatLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Initialize ConvTranspose2D with bilinear.
        self.upsample_edge_side2.weight.data = init_bilinear(self.upsample_edge_side2.weight)
        self.upsample_edge_side3.weight.data = init_bilinear(self.upsample_edge_side3.weight)
        self.upsample_cls_side5.weight.data = init_bilinear(self.upsample_cls_side5.weight)

        # Initialize final conv fusion layer with constant=0.25
        self.ce_fusion.weight.data.fill_(0.25)
        self.ce_fusion.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, special_case=False):
        """
        special case only for res5a branch
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            set_require_grad_to_false(downsample[1])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, special_case=special_case))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, special_case=special_case))

        return nn.Sequential(*layers)

    def forward(self, x, for_vis=False):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu(x) # BS X 64 X 472 X 472
        score_feats1 = self.score_edge_side1(x) # BS X 1 X 472 X 472
        
        x = self.maxpool(x) # BS X 64 X 236 X 236
        
        x = self.res2(x) # BS X 256 X 236 X 236
        score_feats2 = self.score_edge_side2(x) # BS X 1 X 472 X 472
        upsampled_score_feats2 = self.upsample_edge_side2(score_feats2)
        cropped_score_feats2 = upsampled_score_feats2 # Here don't need to crop. (In official caffe, there's crop)

        x = self.res3(x) # BS X 512 X 118 X 118
        score_feats3 = self.score_edge_side3(x) # BS X 1 X 476 X 476
        upsampled_score_feats3 = self.upsample_edge_side3(score_feats3)
        cropped_score_feats3 = self.crop_layer(upsampled_score_feats3, offset=2) # BS X 1 X 472 X 472
        
        x = self.res4(x)
        x = self.res5(x)
        score_feats5 = self.score_cls_side5(x)
        upsampled_score_feats5 = self.upsample_cls_side5(score_feats5)
        cropped_score_feats5 = self.crop_layer(upsampled_score_feats5, offset=4) # BS X 19 X 472 X 472. The output of it will be used to get a loss for this branch.
        sliced_list = self.slice_layer(cropped_score_feats5) # Each element is BS X 1 X 472 X 472
        
        # Add low-level feats to sliced_list
        final_sliced_list = []
        for i in xrange(len(sliced_list)):
            final_sliced_list.append(sliced_list[i])
            final_sliced_list.append(score_feats1)
            final_sliced_list.append(cropped_score_feats2)
            final_sliced_list.append(cropped_score_feats3)

        concat_feats = self.concat_layer(final_sliced_list, dim=1) # BS X 80 X 472 X 472
        fused_feats = self.ce_fusion(concat_feats) # BS X 19 X 472 X 472. The output of this will gen loss for this branch. So, totaly 2 loss. (same loss type)
        
        if for_vis:
            return score_feats1, cropped_score_feats2, cropped_score_feats3, cropped_score_feats5, fused_feats
        else:
            return cropped_score_feats5, fused_feats

def CASENet_resnet101(pretrained=False, num_classes=19):
    """Constructs a modified ResNet-101 model for CASENet.
    Args:
        pretrained (bool): If True, returns a model pre-trained on MSCOCO
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    if pretrained:
        utils.load_official_pretrained_model(model, "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch-cityscapes/pretrained_models/model_casenet.pth.tar") 
    return model

if __name__ == "__main__":
    model = CASENet_resnet101(pretrained=False, num_classes=19) # 19 classes for Cityscapes

    npy_folder = "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch-cityscapes/pretrained_models/"
    loaded_model_path = "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch-cityscapes/pretrained_models/model_casenet.pth.tar"
    layer_to_name_dict = gen_mapping_layer_name(model)
    load_npy_to_layer(model, layer_to_name_dict, npy_folder, loaded_model_path)
    
    # npy_folder = "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch/init_models/"
    # # loaded_model_path = "/Users/anirudhchakravarthy/Documents/CodeArchive/Asian-Paints/CASENet-torch/init_models/casenet_inst_init.pth.tar"
    # layer_to_name_dict = gen_mapping_layer_name(model)
    # load_npy_to_layer(model, layer_to_name_dict, npy_folder, loaded_model_path)
    
    input_data = torch.rand(2, 3, 472, 472)
    input_var = Variable(input_data)
    output1, output2  = model(input_var) 
    print("output1.size:{0}".format(output1.size()))
    print("output2.size:{0}".format(output2.size()))
    feats1, feats2, feats3, feats5, fused_feats = model(input_var, for_vis=True)
    print("feats1.size:{0}".format(feats1.size()))
    print("feats2.size:{0}".format(feats2.size()))
    print("feats3.size:{0}".format(feats3.size()))
    print("feats5.size:{0}".format(feats5.size()))
    print("fused feats.size:{0}".format(fused_feats.size()))

