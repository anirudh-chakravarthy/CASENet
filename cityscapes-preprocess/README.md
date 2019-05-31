# cityscapes-preprocess
Cityscapes Dataset proprocessing code for **CASENet**

### License

The preprocessing code is released under the MIT License (refer to the LICENSE file for details).

### Introduction

The repository contains the preprocessing code of the [Cityscapes dataset](https://www.cityscapes-dataset.com/) for **CASENet**. CASENet is a recently proposed deep network with state of the art performance on category-aware semantic edge detection. For more information about CASENet, please refer to the [arXiv paper](https://arxiv.org/pdf/1705.09759.pdf) and the paper published in [CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_CASENet_Deep_Category-Aware_CVPR_2017_paper.pdf).

### Citation

If you find **CASENet** useful in your research, please consider to cite:

    @inproceedings{yu2017casenet,
        author = {Yu, Zhiding and Feng, Chen and Liu, Ming-Yu and Ramalingam, Srikumar},
        title = {CASENet: Deep Category-Aware Semantic Edge Detection},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        Year = {2017}
    }

    @inproceedings{yu2018seal,
        author = {Yu, Zhiding and Liu, Weiyang and Zou, Yang and Feng, Chen and Ramalingam, Srikumar and Kumar, B. V. K. Vijaya and Kautz, Jan},
        title = {Simultaneous Edge Alignment and Learning},
        booktitle = {Proceedings of the European Conference on Computer Vision},
        Year = {2018}
    }

### Usage
**Note:** In this part, we assume you are in the directory **`$CITYSCAPES_PREPROCESS_ROOT/`**

1. Download the files "gtFine_trainvaltest.zip" and "leftImg8bit_trainvaltest.zip" from the Cityscapes website to **`data_orig/`**, and unzip them.

	```Shell
	unzip data_orig/gtFine_trainvaltest.zip && rm data_orig/gtFine_trainvaltest.zip
	unzip data_orig/leftImg8bit_trainvaltest.zip && rm data_orig/leftImg8bit_trainvaltest.zip
	```
2. Run the matlab code to preprocess the data.

	```Matlab
	# In Matlab Command Window
	run code/demo_preproc.m
	```
    This will generate the .bin edge labels and the corresponding file lists that could be read by CASENet in **`data_proc/`**.

### Related toolkit

The repository of the SBD preprocessing code can be found [here](https://github.com/Chrisding/sbd-preprocess).
