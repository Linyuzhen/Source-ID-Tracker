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
    parser.add_argument('--root_path', type=str, default='/data/linyz/CDF-face')
    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--gpu_id', type=int, default=5)
    args = parser.parse_args()

    return args


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


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

    # load pretrained CosFace model. Borrowing from https://github.com/MuggleWang/CosFace_pytorch
    model = net.sphere().cuda()

    model.load_state_dict(torch.load('/data/linyz/SIDT/ACC99.28.pth'))
    model.train(False)
    model.eval()

    result_txt = open('%s/id_sim_perturbation_celeb.txt' % args.save_path, 'w', encoding='utf-8')
    # JPEG Compression
    for quality in range(40, 91, 10):
        container_id_value = []
        rev_id_value = []
        for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
            rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0],
                                         fake_video_name.split('/')[1])
            # rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)

            root_path = os.path.join(args.root_path, source_video_name)
            image_names = os.listdir(rev_save_path)
            for name in image_names:
                image1 = io.imread(os.path.join(root_path, name))
                image2 = io.imread(
                    os.path.join(rev_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality), name))
                image1 = resize(image1, (112, 96))
                image2 = resize(image2, (112, 96))
                image1 = test_transform(image=image1)["image"]
                image2 = test_transform(image=image2)["image"]
                image1 = image1.unsqueeze(0).cuda()
                image2 = image2.unsqueeze(0).cuda()

                with torch.no_grad():
                    id_value = cosine_sim(model(image1), model(image2))
                rev_id_value.append(id_value.detach().cpu().numpy())

        print('Container: ID Sim: %.4f' % np.mean(container_id_value))
        print('Rev: ID Sim: %.4f' % np.mean(rev_id_value))
        result_txt.write('JPEG Compression: {:g} Container: ID Sim: {:g} Rev: ID Sim: {:g}  \n'.
                         format(quality, np.mean(container_id_value), np.mean(rev_id_value)))

    # Resize
    for scale in [0.8, 0.9, 1.1, 1.2]:
        container_id_value = []
        rev_id_value = []
        for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
            rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0],
                                         fake_video_name.split('/')[1])
            # rev_save_path = os.path.join('%s/images_ffdf' % args.save_path, source_video_name)

            root_path = os.path.join(args.root_path, source_video_name)
            image_names = os.listdir(rev_save_path)
            for name in image_names:
                image1 = io.imread(os.path.join(root_path, name))
                image2 = io.imread(
                    os.path.join(rev_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale), name))
                image1 = resize(image1, (112, 96))
                image2 = resize(image2, (112, 96))
                image1 = test_transform(image=image1)["image"]
                image2 = test_transform(image=image2)["image"]
                image1 = image1.unsqueeze(0).cuda()
                image2 = image2.unsqueeze(0).cuda()

                with torch.no_grad():
                    id_value = cosine_sim(model(image1), model(image2))
                rev_id_value.append(id_value.detach().cpu().numpy())

        print('Container: ID Sim: %.4f' % np.mean(container_id_value))
        print('Rev: ID Sim: %.4f' % np.mean(rev_id_value))
        result_txt.write('Resize: {:g} Container: ID Sim: {:g} Rev: ID Sim: {:g}  \n'.
                         format(scale, np.mean(container_id_value), np.mean(rev_id_value)))

    # Crop
    for scale in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        container_id_value = []
        rev_id_value = []
        for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
            rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0],
                                         fake_video_name.split('/')[1])
            # rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)

            root_path = os.path.join(args.root_path, source_video_name)
            image_names = os.listdir(rev_save_path)
            for name in image_names:
                image1 = io.imread(os.path.join(root_path, name))
                image1 = resize(image1, (256, 256))
                height, width, channel = image1.shape
                new_height = int(height * scale)
                new_width = int(width * scale)
                remaining_height, remaining_width = height - new_height, width - new_width
                remaining_height = int(remaining_height / 2.0)
                remaining_width = int(remaining_width / 2.0)
                image1 = image1[remaining_height:remaining_height + new_height,
                         remaining_width:remaining_width + new_width, :]

                image2 = io.imread(
                    os.path.join(rev_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale), name))
                image1 = resize(image1, (112, 96))
                image2 = resize(image2, (112, 96))
                image1 = test_transform(image=image1)["image"]
                image2 = test_transform(image=image2)["image"]
                image1 = image1.unsqueeze(0).cuda()
                image2 = image2.unsqueeze(0).cuda()

                with torch.no_grad():
                    id_value = cosine_sim(model(image1), model(image2))
                rev_id_value.append(id_value.detach().cpu().numpy())

        print('Container: ID Sim: %.4f' % np.mean(container_id_value))
        print('Rev: ID Sim: %.4f' % np.mean(rev_id_value))
        result_txt.write('Crop: {:g} Container: ID Sim: {:g} Rev: ID Sim: {:g}  \n'.
                         format(scale, np.mean(container_id_value), np.mean(rev_id_value)))

    # Noise
    for rnd_noise in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        container_id_value = []
        rev_id_value = []
        for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
            rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0],
                                         fake_video_name.split('/')[1])
            # rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)

            root_path = os.path.join(args.root_path, source_video_name)
            image_names = os.listdir(rev_save_path)
            for name in image_names:
                image1 = io.imread(os.path.join(root_path, name))
                image2 = io.imread(
                    os.path.join(rev_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise), name))
                image1 = resize(image1, (112, 96))
                image2 = resize(image2, (112, 96))
                image1 = test_transform(image=image1)["image"]
                image2 = test_transform(image=image2)["image"]
                image1 = image1.unsqueeze(0).cuda()
                image2 = image2.unsqueeze(0).cuda()

                with torch.no_grad():
                    id_value = cosine_sim(model(image1), model(image2))
                rev_id_value.append(id_value.detach().cpu().numpy())

        print('Container: ID Sim: %.4f' % np.mean(container_id_value))
        print('Rev: ID Sim: %.4f' % np.mean(rev_id_value))
        result_txt.write('Noise: {:g} Container: ID Sim: {:g} Rev: ID Sim: {:g}  \n'.
                         format(rnd_noise, np.mean(container_id_value), np.mean(rev_id_value)))

    # Brightness
    for contrast_scale in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
        container_id_value = []
        rev_id_value = []
        for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
            rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name.split('/')[0],
                                         fake_video_name.split('/')[1])
            # rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)

            root_path = os.path.join(args.root_path, source_video_name)
            image_names = os.listdir(rev_save_path)
            for name in image_names:
                image1 = io.imread(os.path.join(root_path, name))
                image2 = io.imread(os.path.join(
                    rev_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale), name))
                image1 = resize(image1, (112, 96))
                image2 = resize(image2, (112, 96))
                image1 = test_transform(image=image1)["image"]
                image2 = test_transform(image=image2)["image"]
                image1 = image1.unsqueeze(0).cuda()
                image2 = image2.unsqueeze(0).cuda()

                with torch.no_grad():
                    id_value = cosine_sim(model(image1), model(image2))
                rev_id_value.append(id_value.detach().cpu().numpy())

        print('Container: ID Sim: %.4f' % np.mean(container_id_value))
        print('Rev: ID Sim: %.4f' % np.mean(rev_id_value))
        result_txt.write('Brightness: {:g} Container: ID Sim: {:g} Rev: ID Sim: {:g}  \n'.
                         format(contrast_scale, np.mean(container_id_value), np.mean(rev_id_value)))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    test_ff()
