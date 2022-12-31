import torch
import argparse

import cv2
import random
import numpy as np

from model import RNet
from transforms import build_transforms
from test_data_loader import Test_DataloaderV2
import json
from collections import OrderedDict
from DeepFakeMask import dfl_full, facehull, components, extended

import os


######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ff_df_dataset')

    parser.add_argument('--root_path', type=str, default='/data/linyz/FaceForensics_Fingerprints')
    parser.add_argument('--root_path_celeb_df', type=str, default='/data/linyz/Celeb-DF-v2/face_crop_png')
    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--face_id_save_path', type=str, default='/data/linyz/SIDT/Arcface.pth')

    parser.add_argument('--gpu_id', type=int, default=5)

    parser.add_argument('--rnd_bri_ramp', type=int, default=5000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=5000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=5000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=5000)
    parser.add_argument('--contrast_ramp', type=int, default=5000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=5000)
    parser.add_argument('--rnd_crop_ramp', type=int, default=5000)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_resize_ramp', type=int, default=5000)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_bri', type=float, default=.1)
    parser.add_argument('--rnd_hue', type=float, default=.05)
    parser.add_argument('--jpeg_quality', type=float, default=50)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--contrast_low', type=float, default=.8)
    parser.add_argument('--contrast_high', type=float, default=1.2)
    parser.add_argument('--rnd_sat', type=float, default=0.5)
    parser.add_argument('--blur_prob', type=float, default=0.1)
    parser.add_argument('--no_jpeg', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--rnd_crop', type=float, default=0.2)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_resize', type=float, default=0.2)  # Borrowed from HiDDeN

    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=10.0)
    parser.add_argument('--v_scale', type=float, default=10.0)

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_epochs', type=int, default=4)
    parser.add_argument('--start_val_epochs', type=int, default=0)
    parser.add_argument('--adjust_lr_epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=list, default=[1.0, 1.0, 1.0, 0.9, 0.9, 0.9],
                        help='alpha for L2 Loss of Container, CosSimilarity Loss of Container ID, Lpip,\
                             L2 Loss of Revealed, CosSimilarity Loss of Revealed ID, Lpip')

    args = parser.parse_args()
    return args


def load_landmarks(landmarks_file):
    """

    :param landmarks_file: input landmarks json file name
    :return: all_landmarks: having the shape of 64x2 list. represent left eye,
                            right eye, noise, left lip, right lip
    """
    all_landmarks = OrderedDict()
    with open(landmarks_file, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            line = json.loads(line)
            all_landmarks[line["image_name"]] = np.array(line["landmarks"])
            line = file.readline()
    return all_landmarks


def test_celeb_df():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                       max_pixel_value=255.0, norm_mean=[0, 0, 0],
                                                       norm_std=[1.0, 1.0, 1.0])

    with open('../test_celeb_df_fake.txt', 'r') as f:
        fake_test_videos = f.readlines()
        fake_test_videos = [i.strip() for i in fake_test_videos]

    with open('../test_celeb_df_source.txt', 'r') as f:
        source_test_videos = f.readlines()
        source_test_videos = [i.strip() for i in source_test_videos]

    decoder = RNet().cuda()

    decoder = load_network(decoder, '%s/models/decoder.pth' % args.save_path)

    decoder.train(False)
    decoder.eval()

    for idx, (fake_video_name, source_video_name) in enumerate(zip(fake_test_videos, source_test_videos)):
        aa = fake_video_name.split('/')
        json_file = os.path.join(aa[0], 'dlib_landmarks', aa[1] + '.json')
        json_path = os.path.join(args.save_path, 'images_celeb', json_file)

        fake_frame_path = os.path.join(args.root_path_celeb_df, fake_video_name)
        source_frame_path = os.path.join(args.root_path_celeb_df, source_video_name)
        test_dataset = Test_DataloaderV2(fake_frame_path=fake_frame_path, source_frame_path=source_frame_path,
                                         landmarks_file_path=json_path, phase='test', transform=transform_test,
                                         test_frame_nums=500, size=(args.resolution, args.resolution))

        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.val_batch_size,
                                                  drop_last=False, num_workers=1, pin_memory=True)

        container_save_path = os.path.join('%s/images_celeb' % args.save_path, fake_video_name)
        source_video_name = os.path.join(source_video_name.split('/')[0], fake_video_name.split('/')[1])
        rev_save_path = os.path.join('%s/images_celeb' % args.save_path, source_video_name)

        for fake_image, source_image, face_mask, image_name in test_loader:

            # JPEG Compression            
            for quality in range(40, 91, 10):
                json_path = container_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality).replace(
                    'Celeb-synthesis', 'Celeb-synthesis/dlib_landmarks').replace('.mp4', '.mp4.json')
                all_landmarks = load_landmarks(json_path)

                for j, name in enumerate(image_name):
                    container_fake_image = cv2.imread(
                        os.path.join(container_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality),
                                     name))
                    container_fake_image = cv2.resize(container_fake_image, (256, 256))
                    container_fake_image = cv2.cvtColor(container_fake_image, cv2.COLOR_BGR2RGB) / 255.0
                    container_fake_image = torch.from_numpy(container_fake_image.transpose(2, 0, 1)).unsqueeze(
                        0).cuda().detach().float()
                    try:
                        landmarks = all_landmarks[name].astype('int32')
                        face_mask = facehull(landmarks=landmarks,
                                             face=np.ones((256, 256, 3)), channels=3).mask / 255.0
                    except:
                        print(json_path)
                        face_mask = np.ones((256, 256, 3))
                    face_mask = torch.from_numpy(face_mask.transpose(2, 0, 1)).unsqueeze(0).cuda().detach().float()

                    with torch.no_grad():
                        rev_source_image = decoder(container_fake_image,
                                                face_mask)  # put concatenated image into R-net and get revealed secret image
                        rev_source_image = rev_source_image.cpu().numpy()

                    if not os.path.isdir(rev_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality)):
                        os.makedirs(rev_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality))

                    cv2.imwrite(
                        os.path.join(rev_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality), name),
                        cv2.cvtColor((rev_source_image[0].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

            # Resize
            # for scale in [0.8, 0.9, 1.1, 1.2]:
            #     for j, name in enumerate(image_name):
            #         container_fake_image = cv2.imread(os.path.join(container_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale), name))
            #         container_fake_image = cv2.resize(container_fake_image, (256, 256))
            #         container_fake_image = cv2.cvtColor(container_fake_image, cv2.COLOR_BGR2RGB)/255.0
            #         container_fake_image = torch.from_numpy(container_fake_image.transpose(2,0,1)).unsqueeze(0).cuda().detach().float()
            #         with torch.no_grad():
            #             rev_source_image = decoder(container_fake_image, face_mask[j:j+1])  # put concatenated image into R-net and get revealed secret image
            #             rev_source_image = rev_source_image.cpu().numpy()
            #
            #
            #         if not os.path.isdir(rev_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale)):
            #             os.makedirs(rev_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale))
            #
            #         cv2.imwrite(os.path.join(rev_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale), name),
            #                     cv2.cvtColor((rev_source_image[0].transpose(1,2,0)*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # Crop
            for scale in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                json_path = container_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale).replace(
                    'Celeb-synthesis', 'Celeb-synthesis/dlib_landmarks').replace('.mp4', '.mp4.json')
                all_landmarks = load_landmarks(json_path)
                for j, name in enumerate(image_name):
                    container_fake_image = cv2.imread(
                        os.path.join(container_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale),
                                     name))
                    container_fake_image = cv2.resize(container_fake_image, (256, 256))
                    container_fake_image = cv2.cvtColor(container_fake_image, cv2.COLOR_BGR2RGB) / 255.0
                    container_fake_image = torch.from_numpy(container_fake_image.transpose(2, 0, 1)).unsqueeze(
                        0).cuda().detach().float()
                    try:
                        landmarks = all_landmarks[name].astype('int32')
                        face_mask = facehull(landmarks=landmarks,
                                             face=np.ones((256, 256, 3)), channels=3).mask / 255.0
                    except:
                        print(json_path)
                        face_mask = np.ones((256, 256, 3))
                    face_mask = torch.from_numpy(face_mask.transpose(2, 0, 1)).unsqueeze(0).cuda().detach().float()
                    with torch.no_grad():
                        rev_source_image = decoder(container_fake_image,
                                                face_mask)  # put concatenated image into R-net and get revealed secret image
                        rev_source_image = rev_source_image.cpu().numpy()

                    if not os.path.isdir(rev_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale)):
                        os.makedirs(rev_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale))

                    cv2.imwrite(
                        os.path.join(rev_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale), name),
                        cv2.cvtColor((rev_source_image[0].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

            for rnd_noise in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
                json_path = container_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise).replace(
                    'Celeb-synthesis', 'Celeb-synthesis/dlib_landmarks').replace('.mp4', '.mp4.json')
                all_landmarks = load_landmarks(json_path)
                for j, name in enumerate(image_name):
                    container_fake_image = cv2.imread(
                        os.path.join(container_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise),
                                     name))
                    container_fake_image = cv2.resize(container_fake_image, (256, 256))
                    container_fake_image = cv2.cvtColor(container_fake_image, cv2.COLOR_BGR2RGB) / 255.0
                    container_fake_image = torch.from_numpy(container_fake_image.transpose(2, 0, 1)).unsqueeze(
                        0).cuda().detach().float()
                    try:
                        landmarks = all_landmarks[name].astype('int32')
                        face_mask = facehull(landmarks=landmarks,
                                             face=np.ones((256, 256, 3)), channels=3).mask / 255.0
                    except:
                        print(json_path)
                        face_mask = np.ones((256, 256, 3))
                    face_mask = torch.from_numpy(face_mask.transpose(2, 0, 1)).unsqueeze(0).cuda().detach().float()
                    with torch.no_grad():
                        rev_source_image = decoder(container_fake_image,
                                                face_mask)  # put concatenated image into R-net and get revealed secret image
                        rev_source_image = rev_source_image.cpu().numpy()

                    if not os.path.isdir(rev_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise)):
                        os.makedirs(rev_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise))

                    cv2.imwrite(
                        os.path.join(rev_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise),
                                     name),
                        cv2.cvtColor((rev_source_image[0].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

            for contrast_scale in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
                json_path = container_save_path.replace('save_result',
                                                        'save_result/Brightness/Scale_%f' % contrast_scale).replace(
                    'Celeb-synthesis', 'Celeb-synthesis/dlib_landmarks').replace('.mp4', '.mp4.json')
                all_landmarks = load_landmarks(json_path)
                for j, name in enumerate(image_name):
                    container_fake_image = cv2.imread(os.path.join(
                        container_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale),
                        name))
                    container_fake_image = cv2.resize(container_fake_image, (256, 256))
                    container_fake_image = cv2.cvtColor(container_fake_image, cv2.COLOR_BGR2RGB) / 255.0
                    container_fake_image = torch.from_numpy(container_fake_image.transpose(2, 0, 1)).unsqueeze(
                        0).cuda().detach().float()
                    try:
                        landmarks = all_landmarks[name].astype('int32')
                        face_mask = facehull(landmarks=landmarks,
                                             face=np.ones((256, 256, 3)), channels=3).mask / 255.0
                    except:
                        print(json_path)
                        face_mask = np.ones((256, 256, 3))
                    face_mask = torch.from_numpy(face_mask.transpose(2, 0, 1)).unsqueeze(0).cuda().detach().float()
                    with torch.no_grad():
                        rev_source_image = decoder(container_fake_image,
                                                face_mask)  # put concatenated image into R-net and get revealed secret image
                        rev_source_image = rev_source_image.cpu().numpy()

                    if not os.path.isdir(
                            rev_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale)):
                        os.makedirs(
                            rev_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale))

                    cv2.imwrite(os.path.join(
                        rev_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale), name),
                                cv2.cvtColor((rev_source_image[0].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                             cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # if not os.path.exists('%s/images_df' % args.save_path):
    #     os.makedirs('%s/images_df' % args.save_path)
    if not os.path.exists('%s/images_celeb' % args.save_path):
        os.makedirs('%s/images_celeb' % args.save_path)
    # test_ff()
    test_celeb_df()
