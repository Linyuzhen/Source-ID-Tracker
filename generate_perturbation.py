from tqdm import tqdm
import numpy as np
import copy
import argparse
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Generating distortions for images')

    parser.add_argument('--root_path', type=str, default='/data/linyz/FF-face')
    # parser.add_argument('--root_path', type=str, default='/data/linyz/CDF-face')

    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--gpu_id', type=int, default=1)
    args = parser.parse_args()

    return args


def gen_ds():
    args = parse_args()
    # with open('../test_df_fake_c23.txt', 'r') as f:
    with open('./test_celeb_df_fake.txt', 'r') as f:
        fake_test_videos = f.readlines()
        fake_test_videos = [i.strip() for i in fake_test_videos]

    # with open('../test_df_source_c23.txt', 'r') as f:
    with open('./test_celeb_df_source.txt', 'r') as f:
        source_test_videos = f.readlines()
        source_test_videos = [i.strip() for i in source_test_videos]

    for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
        container_save_path = os.path.join('%s/images_celeb' % args.save_path, fake_video_name)
        # container_save_path = os.path.join('%s/images_df' % args.save_path, fake_video_name)
        image_names = os.listdir(container_save_path)
        for name in image_names:
            image = cv2.imread(os.path.join(container_save_path, name))

            # JPEG Compression
            for quality in range(40, 91, 10):
                image_temp = copy.deepcopy(image)
                if not os.path.isdir(container_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality)):
                    os.makedirs(container_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                face_img_encode = cv2.imencode('.jpg', image_temp, encode_param)[1]
                image_qf = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
                cv2.imwrite(
                    os.path.join(container_save_path.replace('save_result', 'save_result/JPEG/qf_%d' % quality), name),
                    image_qf)

            # Resize
            for scale in [0.8, 0.9, 1.1, 1.2]:
                image_temp = copy.deepcopy(image)
                if not os.path.isdir(container_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale)):
                    os.makedirs(container_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale))
                height, width, channel = image_temp.shape
                image_scale = cv2.resize(image_temp, (int(width * scale), int(height * scale)))
                cv2.imwrite(
                    os.path.join(container_save_path.replace('save_result', 'save_result/Resize/Scale_%f' % scale),
                                 name), image_scale)

            # Crop
            for scale in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                image_temp = copy.deepcopy(image)
                if not os.path.isdir(container_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale)):
                    os.makedirs(container_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale))
                height, width, channel = image_temp.shape
                new_height = int(height * scale)
                new_width = int(width * scale)
                remaining_height, remaining_width = height - new_height, width - new_width
                remaining_height = int(remaining_height / 2.0)
                remaining_width = int(remaining_width / 2.0)
                image_crop = image_temp[remaining_height:remaining_height + new_height,
                             remaining_width:remaining_width + new_width, :]
                cv2.imwrite(
                    os.path.join(container_save_path.replace('save_result', 'save_result/Crop/Scale_%f' % scale), name),
                    image_crop)

            for rnd_noise in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
                image_temp = copy.deepcopy(image)
                if not os.path.isdir(
                        container_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise)):
                    os.makedirs(container_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise))
                image_temp = image_temp / 255.0
                noise = np.random.normal(loc=0, scale=rnd_noise, size=image_temp.shape).astype(np.float32)
                image_temp = image_temp + noise
                image_temp = np.clip(image_temp, 0, 1.0)
                image_temp = (image_temp * 255.0).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(container_save_path.replace('save_result', 'save_result/Noise/Rnd_%f' % rnd_noise),
                                 name), image_temp)

            for contrast_scale in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
                image_temp = copy.deepcopy(image)
                if not os.path.isdir(
                        container_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale)):
                    os.makedirs(
                        container_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale))
                image_temp = image_temp / 255.0
                image_temp = image_temp * contrast_scale
                image_temp = np.clip(image_temp, 0, 1.0)
                image_temp = (image_temp * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(
                    container_save_path.replace('save_result', 'save_result/Brightness/Scale_%f' % contrast_scale),
                    name), image_temp)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    gen_ds()
