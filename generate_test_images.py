import torch
from torch.autograd import Variable
import argparse

import cv2
import numpy as np

from model import ENet, RNet, dsl_net
from transforms import build_transforms
from test_data_loader import Test_Dataloader

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
    parser = argparse.ArgumentParser(description='Generating test images')

    parser.add_argument('--root_path', type=str, default='/data/linyz/FaceForensics_Fingerprints')
    parser.add_argument('--root_path_celeb_df', type=str, default='/data/linyz/Celeb-DF-v2/face_crop_png')
    parser.add_argument('--save_path', type=str, default='./save_result')
    # parser.add_argument('--pretrained', type=str, default='/home/linyz/UCL_Blending/xception-43020ad28.pth')     
    parser.add_argument('--face_id_save_path', type=str, default='/data/linyz/SIDT/Arcface.pth')

    parser.add_argument('--gpu_id', type=int, default=0)

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


def test_ff():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                       max_pixel_value=255.0, norm_mean=[0, 0, 0],
                                                       norm_std=[1.0, 1.0, 1.0])

    with open('../test_df_fake_c23.txt', 'r') as f:
        fake_test_videos = f.readlines()
        fake_test_videos = [i.strip() for i in fake_test_videos]

    with open('../test_df_source_c23.txt', 'r') as f:
        source_test_videos = f.readlines()
        source_test_videos = [i.strip() for i in source_test_videos]

    encoder = ENet().cuda()
    decoder = RNet().cuda()

    encoder = load_network(encoder, '%s/models/encoder.pth' % args.save_path)
    decoder = load_network(decoder, '%s/models/decoder.pth' % args.save_path)

    encoder.train(False)
    decoder.train(False)
    encoder.eval()
    decoder.eval()

    for idx, (fake_video_name, source_video_name) in enumerate(zip(fake_test_videos, source_test_videos)):
        aa = source_video_name.split('/')
        json_file = os.path.join(aa[0], aa[1], aa[2], aa[3], 'dlib_landmarks', aa[4] + '.json')
        json_path = os.path.join(args.root_path, json_file)

        fake_frame_path = os.path.join(args.root_path, fake_video_name)
        source_frame_path = os.path.join(args.root_path, source_video_name)
        test_dataset = Test_Dataloader(fake_frame_path=fake_frame_path, source_frame_path=source_frame_path,
                                       landmarks_file_path=json_path, phase='test', transform=transform_test,
                                       test_frame_nums=500, size=(args.resolution, args.resolution))

        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.val_batch_size,
                                                  drop_last=False, num_workers=1, pin_memory=True)

        for fake_image, source_image, face_mask, image_name in test_loader:
            # wrap them in Variable
            fake_image = Variable(fake_image.cuda().detach())
            source_image = Variable(source_image.cuda().detach())
            face_mask = Variable(face_mask.cuda().detach())

            with torch.no_grad():
                container_fake_image = encoder(source_image, fake_image, face_mask)
                rev_source_image = RNet(container_fake_image,
                                        face_mask)  # put concatenated image into R-net and get revealed secret image

                container_fake_image = container_fake_image.cpu().numpy()
                rev_source_image = rev_source_image.cpu().numpy()
            for j, name in enumerate(image_name):
                container_save_path = os.path.join('%s/images_df' % args.save_path, fake_video_name)
                temp_image_root_path = '%s/images_df' % args.save_path
                for temp_name in fake_video_name.split('/'):
                    temp_image_root_path = os.path.join(temp_image_root_path, temp_name)
                    if not os.path.isdir(temp_image_root_path):
                        os.mkdir(temp_image_root_path)

                cv2.imwrite(os.path.join(container_save_path, name),
                            cv2.cvtColor((container_fake_image[j].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))

                rev_save_path = os.path.join('%s/images_df' % args.save_path, source_video_name)
                temp_image_root_path = '%s/images_df' % args.save_path
                for temp_name in source_video_name.split('/'):
                    temp_image_root_path = os.path.join(temp_image_root_path, temp_name)
                    if not os.path.isdir(temp_image_root_path):
                        os.mkdir(temp_image_root_path)

                cv2.imwrite(os.path.join(rev_save_path, name),
                            cv2.cvtColor((rev_source_image[j].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))


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

    encoder = ENet().cuda()
    decoder = RNet().cuda()

    encoder = load_network(encoder, '%s/models/encoder.pth' % args.save_path)
    decoder = load_network(decoder, '%s/models/decoder.pth' % args.save_path)

    encoder.train(False)
    decoder.train(False)
    encoder.eval()
    decoder.eval()

    for idx, (fake_video_name, source_video_name) in enumerate(zip(fake_test_videos, source_test_videos)):
        aa = source_video_name.split('/')
        json_file = os.path.join(aa[0], 'dlib_landmarks', aa[1] + '.json')
        json_path = os.path.join(args.root_path_celeb_df, json_file)

        fake_frame_path = os.path.join(args.root_path_celeb_df, fake_video_name)
        source_frame_path = os.path.join(args.root_path_celeb_df, source_video_name)
        test_dataset = Test_Dataloader(fake_frame_path=fake_frame_path, source_frame_path=source_frame_path,
                                       landmarks_file_path=json_path, phase='test', transform=transform_test,
                                       test_frame_nums=500, size=(args.resolution, args.resolution))

        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.val_batch_size,
                                                  drop_last=False, num_workers=1, pin_memory=True)

        for fake_image, source_image, face_mask, image_name in test_loader:
            # wrap them in Variable
            fake_image = Variable(fake_image.cuda().detach())
            source_image = Variable(source_image.cuda().detach())
            face_mask = Variable(face_mask.cuda().detach())

            with torch.no_grad():
                container_fake_image = encoder(source_image, fake_image, face_mask)
                rev_source_image = decoder(container_fake_image,
                                           face_mask)  # put concatenated image into R-net and get revealed secret image

                container_fake_image = container_fake_image.cpu().numpy()
                rev_source_image = rev_source_image.cpu().numpy()
            for j, name in enumerate(image_name):
                container_save_path = os.path.join('%s/images_celeb_df' % args.save_path, fake_video_name)
                temp_image_root_path = '%s/images_celeb_df' % args.save_path
                for temp_name in fake_video_name.split('/'):
                    temp_image_root_path = os.path.join(temp_image_root_path, temp_name)
                    if not os.path.isdir(temp_image_root_path):
                        os.mkdir(temp_image_root_path)

                cv2.imwrite(os.path.join(container_save_path, name),
                            cv2.cvtColor((container_fake_image[j].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))

                source_video_name = os.path.join(source_video_name.split('/')[0], fake_video_name.split('/')[1])
                rev_save_path = os.path.join('%s/images_celeb_df' % args.save_path, source_video_name)
                temp_image_root_path = '%s/images_celeb_df' % args.save_path
                for temp_name in source_video_name.split('/'):
                    temp_image_root_path = os.path.join(temp_image_root_path, temp_name)
                    if not os.path.isdir(temp_image_root_path):
                        os.mkdir(temp_image_root_path)

                cv2.imwrite(os.path.join(rev_save_path, name),
                            cv2.cvtColor((rev_source_image[j].transpose(1, 2, 0) * 255.0).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists('%s/images_df' % args.save_path):
        os.makedirs('%s/images_df' % args.save_path)
    if not os.path.exists('%s/images_celeb_df' % args.save_path):
        os.makedirs('%s/images_celeb_df' % args.save_path)
    test_ff()
    test_celeb_df()
