import skimage.io as io
from skimage.transform import resize
from tqdm import tqdm
import numpy as np 
import lpips
import argparse
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ff_df_dataset')
           
    parser.add_argument('--root_path', type=str, default='/data/linyz/FaceForensics_Fingerprints')           
    # parser.add_argument('--root_path', type=str, default='/data/linyz/Celeb-DF-v2/face_crop_png')           
    parser.add_argument('--save_path', type=str, default='./save_result')     
    parser.add_argument('--gpu_id', type=int, default=6)
    args = parser.parse_args()
    
    return args

def test_ff():
    args = parse_args()
    with open('../test_df_fake_c23.txt', 'r') as f:
    # with open('../test_celeb_df_fake.txt', 'r') as f:
        fake_test_videos = f.readlines()
        fake_test_videos = [i.strip() for i in fake_test_videos]

    with open('../test_df_source_c23.txt', 'r') as f:
    # with open('../test_celeb_df_source.txt', 'r') as f:
        source_test_videos = f.readlines()
        source_test_videos = [i.strip() for i in source_test_videos]

    for idx, (fake_video_name, source_video_name) in tqdm(enumerate(zip(fake_test_videos, source_test_videos))):
        # container_save_path = os.path.join('%s/images_celeb_df' % args.save_path, fake_video_name)
        container_save_path = os.path.join('%s/images_ffdf' % args.save_path, fake_video_name)

        root_path = os.path.join(args.root_path, fake_video_name)
        image_names = os.listdir(container_save_path)
        for name in image_names:
            image1 = cv2.imread(os.path.join(root_path, name))
            image1 = cv2.resize(image1, (256, 256)).astype(np.float32)
            image2 = cv2.imread(os.path.join(container_save_path, name))
            image2 = cv2.resize(image2, (256, 256)).astype(np.float32)
            temp = np.abs(image1 - image2)
            if not os.path.isdir(container_save_path.replace('images_ffdf', 'images_ffdf_diff')):
                os.makedirs(container_save_path.replace('images_ffdf', 'images_ffdf_diff'))

            cv2.imwrite(os.path.join(container_save_path.replace('images_ffdf', 'images_ffdf_diff'), name), temp.astype(np.uint8))


        # rev_save_path = os.path.join('%s/images_celeb_df' % args.save_path, source_video_name.split('/')[0], fake_video_name.split('/')[1])
        rev_save_path = os.path.join('%s/images_ffdf' % args.save_path, source_video_name)

        root_path = os.path.join(args.root_path, source_video_name)
        image_names = os.listdir(rev_save_path)
        for name in image_names:
            image1 = cv2.imread(os.path.join(root_path, name))
            image1 = cv2.resize(image1, (256, 256)).astype(np.float32)
            image2 = cv2.imread(os.path.join(rev_save_path, name))
            image2 = cv2.resize(image2, (256, 256)).astype(np.float32)
            temp = np.abs(image1 - image2)
            if not os.path.isdir(rev_save_path.replace('images_ffdf', 'images_ffdf_diff')):
                os.makedirs(rev_save_path.replace('images_ffdf', 'images_ffdf_diff'))

            cv2.imwrite(os.path.join(rev_save_path.replace('images_ffdf', 'images_ffdf_diff'), name), temp.astype(np.uint8))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    test_ff()

    





