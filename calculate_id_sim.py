"""
Author: HanChen
Date: 11.09.2021
"""
# -*- coding: utf-8 -*-
import skimage.io as io
from skimage.transform import resize
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet101
from tqdm import tqdm
import numpy as np 
import torch 
import argparse
import os

import net

def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ff_df_dataset')
           
    # parser.add_argument('--root_path', type=str, default='/data/linyz/FaceForensics')
    parser.add_argument('--root_path', type=str, default='/data/linyz/Celeb-DF-v2/face_crop_png')
    parser.add_argument('--save_path', type=str, default='./save_result')     
    parser.add_argument('--gpu_id', type=int, default=6)
    args = parser.parse_args()
    
    return args

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

def test_ff():
    args = parse_args()

    # with open('../test_df_fake_c23.txt', 'r') as f:
    with open('../test_celeb_df_fake.txt', 'r') as f:
        fake_test_videos = f.readlines()
        fake_test_videos = [i.strip() for i in fake_test_videos]

    # with open('../test_df_source_c23.txt', 'r') as f:
    with open('../test_celeb_df_source.txt', 'r') as f:
        source_test_videos = f.readlines()
        source_test_videos = [i.strip() for i in source_test_videos]

    test_transform = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0),
            # A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=1.0),
            ToTensorV2()
        ])

    model = net.sphere().cuda()
    # model = resnet101(num_classes=256).cuda()
    # model.load_state_dict(torch.load('/data/linyz/SIDT/Arcface.pth'))

    # Load the pre-trained CosFace model
    model.load_state_dict(torch.load('/data/linyz/SIDT/ACC99.28.pth'))
    model.train(False)
    model.eval()
    
    result = open('celeb_id_cos.txt', 'w')

    container_id_value = []
    rev_id_value = []
    for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):

        container_save_path = os.path.join('%s/images_celeb' % args.save_path, fake_video_name)
        # container_save_path = os.path.join('%s/images_df' % args.save_path, fake_video_name)

        root_path = os.path.join(args.root_path, fake_video_name)
        image_names = os.listdir(container_save_path)
        for name in image_names:
            image1 = io.imread(os.path.join(root_path, name))
            image2 = io.imread(os.path.join(container_save_path, name))
            image1 = resize(image1, (112, 96))
            image2 = resize(image2, (112, 96))
            image1 = test_transform(image=image1)["image"]
            image2 = test_transform(image=image2)["image"]
            image1 = image1.unsqueeze(0).cuda()
            image2 = image2.unsqueeze(0).cuda()
            
            # inner_product = (torch.bmm(F.normalize(model(image1)).detach().unsqueeze(1), F.normalize(model(image2)).detach().unsqueeze(2)).squeeze())
            # id_value = torch.mean(inner_product)
            with torch.no_grad():
                id_value = cosine_sim(model(image1), model(image2))
            container_id_value.append(id_value.detach().cpu().numpy())
            result.write(os.path.join(container_save_path, name) + '%.4f' % id_value.detach().cpu().numpy() + '\n')

        rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0], fake_video_name.split('/')[1])
        # rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)

        root_path = os.path.join(args.root_path, source_video_name)
        image_names = os.listdir(rev_save_path)
        for name in image_names:
            image1 = io.imread(os.path.join(root_path, name))
            image2 = io.imread(os.path.join(rev_save_path, name))
            image1 = resize(image1, (112, 96))
            image2 = resize(image2, (112, 96))
            image1 = test_transform(image=image1)["image"]
            image2 = test_transform(image=image2)["image"]
            image1 = image1.unsqueeze(0).cuda()
            image2 = image2.unsqueeze(0).cuda()
            
            # inner_product = (torch.bmm(F.normalize(model(image1)).detach().unsqueeze(1), F.normalize(model(image2)).detach().unsqueeze(2)).squeeze())
            # id_value = torch.mean(inner_product)
            with torch.no_grad():
                id_value = cosine_sim(model(image1), model(image2))
            rev_id_value.append(id_value.detach().cpu().numpy())
            result.write(os.path.join(rev_save_path, name) + '%.4f' % id_value.detach().cpu().numpy() + '\n')
    print('Container: ID Sim: %.4f' % np.mean(container_id_value) )
    print('Rev: ID Sim: %.4f' % np.mean(rev_id_value) )



if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    test_ff()

    





