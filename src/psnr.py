# -*- coding:utf-8 -*-
"""
Compute PSNR and SSIM with Set12.
"""
import os
import glob
import cv2
import numpy as np
from skimage.measure import compare_ssim
import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.color as sc
import numpy as np
import scipy.misc as misc
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

def calc_PSNR(input, target, set_name='Set5', rgb_range=255, scale=4):
    def quantize(img, rgb_range):
        return np.floor( np.clip(img*(255 / rgb_range)+0.5, 0, 255) )/255
    def rgb2ycbcrT(rgb):
        return sc.rgb2ycbcr(rgb) / 255
    
    test_Y = ['Set5', 'Set14', 'B100', 'Urban100']

    h, w, c = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[0:h, 0:w, :], rgb_range)
    diff = input - target
    if set_name in test_Y:
        shave = scale
        if c > 1:
            input_Y = rgb2ycbcrT(input)
            target_Y = rgb2ycbcrT(target)
            diff = np.reshape(input_Y - target_Y,( h, w, 3))
    else:
        shave = scale + 6

    diff = diff[shave:(h - shave), shave:(w - shave), :]
    mse = np.power(diff,2).mean()
    psnr = -10 * np.log10(mse)

    return psnr

if __name__ == '__main__':
    #D:\wbs\yuv420\downsample\x0\encode\BQMall_832x480_60\22\bmp
    data_set12 = glob.glob(os.path.join("E:/ywj/HTM_dataset/Bookarrival/Renderer/split/", "*.bmp"))
    # data_set12 = glob.glob(os.path.join("D:/wbs/yuv420/ori/bqmall100/", "*.bmp"))
    #data_set12 = glob.glob(os.path.join("D:/wbs/yuv420/downsample/x0/bi/1/", "*.bmp"))

    # data_set2_quality10 = glob.glob(os.path.join("F:\mix/20191127\mixed1\experiment\BQMall37/results-Demo/", "*.bmp"))
    data_set2_quality10 = glob.glob(os.path.join("E:/3DTestSequences/bookarrival/split/", "*.bmp"))

    compress_avg_psnr = 0.
    deblocking_avg_psnr = 0.
    compress_avg_ssim = 0.
    deblocking_avg_ssim = 0.
    for i in range(100):
        #print(len(data_set12))
		# reszie 256 * 256.
        img_set12 = cv2.resize(cv2.imread(str(data_set12[i]), 0), (256, 256))
        img_set12_q10 = cv2.resize(cv2.imread(str(data_set2_quality10[i]), 0), (256, 256))
  
        # label, noisy_image
        psnr_compress = calc_PSNR(cv2.imread(str(data_set2_quality10[i])),cv2.imread(str(data_set12[i])))
        #print(psnr_compress)
        compress_avg_psnr += psnr_compress


        ssim_compress = compare_ssim(img_set12, img_set12_q10)
        #print(ssim_compress)
        compress_avg_ssim += ssim_compress

    print("Average compress PSNR is: {}".format(compress_avg_psnr / 100))
    print("Average compress SSIM is: {}".format(compress_avg_ssim / 100))