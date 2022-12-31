"""
Author: HanChen
Date: 11.09.2021
"""
# -*- coding: utf-8 -*-
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
import lpips
import torch
import numpy as np


def ssim(image1, image2, data_range=255):
    return structural_similarity(image1, image2, multichannel=True, data_range=data_range)


def psnr(image1, image2, data_range=255):
    return peak_signal_noise_ratio(image1, image2, data_range=data_range)


def perceptual_loss(image1, image2, vgg):
    image1 = image1.astype(np.float32).transpose(2, 0, 1) / 255.0
    image2 = image2.astype(np.float32).transpose(2, 0, 1) / 255.0
    with torch.no_grad():
        loss = vgg(torch.from_numpy(np.expand_dims(image1, axis=0)).cuda(),
                   torch.from_numpy(np.expand_dims(image2, axis=0)).cuda())
    return loss


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    image_names = os.listdir('test_images/953.mp4')
    loss_fn_vgg = lpips.LPIPS(net='vgg').eval().cuda()
    for name in image_names:
        image1 = io.imread(os.path.join('test_images/953.mp4', name))
        image2 = io.imread(os.path.join('test_images/953_974.mp4', name))
        image1 = resize(image1, (256, 256)) * 255.0
        image2 = resize(image2, (256, 256)) * 255.0
        ssim_value = ssim(image1, image2, data_range=255)
        psnr_value = psnr(image1, image2, data_range=255)
        lpips_value = perceptual_loss(image1, image2, loss_fn_vgg)
        print('SSIM value:  %.4f' % ssim_value)
        print('PSNR value:  %.4f' % psnr_value)
        print('LPIPS value: %.4f' % lpips_value)
