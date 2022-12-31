"""
Author: HanChen
Date: 11.09.2021
"""
# -*- coding: utf-8 -*-
import skimage.io as io
from skimage.transform import resize
from tqdm import tqdm
import numpy as np 
import lpips
import argparse
import os

from metric import ssim, psnr, perceptual_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ff_df_dataset')
           
    # parser.add_argument('--root_path', type=str, default='/data/linyz/FaceForensics')
    parser.add_argument('--root_path', type=str, default='/data/linyz/Celeb-DF-v2/face_crop_png')

    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--gpu_id', type=int, default=6)
    args = parser.parse_args()
    
    return args

def test_ff():
    args = parse_args()
    loss_fn_vgg = lpips.LPIPS(net='vgg').eval().cuda()

    # with open('../test_df_fake_c23.txt', 'r') as f:
    with open('../test_celeb_df_fake.txt', 'r') as f:
        fake_test_videos = f.readlines()
        fake_test_videos = [i.strip() for i in fake_test_videos]

    # with open('../test_df_source_c23.txt', 'r') as f:
    with open('../test_celeb_df_source.txt', 'r') as f:
        source_test_videos = f.readlines()
        source_test_videos = [i.strip() for i in source_test_videos]

    container_ssim_value = []
    container_psnr_value = []
    container_lpips_value = []
    rev_ssim_value = []
    rev_psnr_value = []
    rev_lpips_value = []
    for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
        container_save_path = os.path.join('%s/images_celeb' % args.save_path, fake_video_name)
        # container_save_path = os.path.join('%s/images_df' % args.save_path, fake_video_name)


        root_path = os.path.join(args.root_path, fake_video_name)
        image_names = os.listdir(container_save_path)
        for name in image_names:
            image1 = io.imread(os.path.join(root_path, name))
            image2 = io.imread(os.path.join(container_save_path, name))
            image1 = resize(image1, (256, 256))*255.0
            image2 = resize(image2, (256, 256))*255.0
            ssim_value = ssim(image1, image2, data_range=255)
            psnr_value = psnr(image1, image2, data_range=255)
            lpips_value = perceptual_loss(image1, image2, loss_fn_vgg)
            container_ssim_value.append(ssim_value)
            container_psnr_value.append(psnr_value)
            container_lpips_value.append(lpips_value.cpu().numpy())


        rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0], fake_video_name.split('/')[1])
        # rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)


        root_path = os.path.join(args.root_path, source_video_name)
        image_names = os.listdir(rev_save_path)
        for name in image_names:
            image1 = io.imread(os.path.join(root_path, name))
            image2 = io.imread(os.path.join(rev_save_path, name))
            image1 = resize(image1, (256, 256))*255.0
            image2 = resize(image2, (256, 256))*255.0
            ssim_value = ssim(image1, image2, data_range=255)
            psnr_value = psnr(image1, image2, data_range=255)
            lpips_value = perceptual_loss(image1, image2, loss_fn_vgg)
            rev_ssim_value.append(ssim_value)
            rev_psnr_value.append(psnr_value)
            rev_lpips_value.append(lpips_value.cpu().numpy())

    print('Container: SSIM: %.4f, PSNR: %.4f, LPIPS %.4f' % (np.mean(container_ssim_value), np.mean(container_psnr_value), np.mean(container_lpips_value)) )
    print('Rev: SSIM: %.4f, PSNR: %.4f, LPIPS %.4f' % (np.mean(rev_ssim_value), np.mean(rev_psnr_value), np.mean(rev_lpips_value)) )



if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    test_ff()

    





