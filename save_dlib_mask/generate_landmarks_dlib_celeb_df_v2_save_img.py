from tqdm import tqdm
from generate_landmarks_dlib import save_landmarks
import os


def main():
    # image_root_path = '/data/linyz/SIDT/HSID_face_loss_yuv_lpip_ffdf11/save_result_wolpips/images_celeb'
    image_root_path = '/data/linyz/SIDT/images_celeb'

    sub_folders = ['Celeb-synthesis']
    for sub_folder in sub_folders:
        image_path = os.path.join(image_root_path, sub_folder)
        save_dir = os.path.join(image_path, 'dlib_landmarks')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        for index, video in tqdm(enumerate(os.listdir(image_path))):
            input_dir = os.path.join(image_path, video)

            save_landmarks(input_dir, save_dir)


if __name__ == "__main__":
    main()




















