import os
import json
import random
from tqdm import tqdm


def ff_main():
    real_path = 'original_sequences/youtube/c23/videos'
    fake_path = 'manipulated_sequences/Deepfakes/c23/videos'
    f = open('splits/test.json', 'r')
    test_json = json.load(f)
    test_videos = []
    for video_name in test_json:
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '.mp4')
        test_videos.append(input_video_path)
        input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        test_videos.append(input_video_path)

        input_video_path = os.path.join(real_path, video_name[0] + '.mp4')
        test_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        test_videos.append(input_video_path)

    f = open('splits/train.json', 'r')
    train_json = json.load(f)
    train_videos = []
    for video_name in train_json:
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '.mp4')
        train_videos.append(input_video_path)
        input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        train_videos.append(input_video_path)

        input_video_path = os.path.join(real_path, video_name[0] + '.mp4')
        train_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        train_videos.append(input_video_path)

    f = open('splits/val.json', 'r')
    val_json = json.load(f)
    val_videos = []
    for video_name in val_json:
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '.mp4')
        val_videos.append(input_video_path)
        input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        val_videos.append(input_video_path)

        input_video_path = os.path.join(real_path, video_name[0] + '.mp4')
        val_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        val_videos.append(input_video_path)

    train_txt = open('../save_txt/train_ff.txt', 'w')
    val_txt = open('../save_txt/val_ff.txt', 'w')
    test_txt = open('../save_txt/test_ff.txt', 'w')
    for i in range(len(train_videos)):
        train_txt.write(train_videos[i] + '\n')
    for i in range(len(val_videos)):
        val_txt.write(val_videos[i] + '\n')
    for i in range(len(test_videos)):
        test_txt.write(test_videos[i] + '\n')


def cdf_main():
    video_root_path = '/home/linyz/datasets/CDF-face'
    txt_name = 'List_of_testing_videos.txt'
    sub_folders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']

    with open(os.path.join(video_root_path, txt_name), 'r') as f:
        test_videos = f.readlines()
        test_videos = [i.strip().split(' ')[1] for i in test_videos]

    train_videos = []
    for sub_folder in sub_folders:
        sub_train_videos = []
        video_path = os.path.join(video_root_path, sub_folder)
        for index, video in tqdm(enumerate(os.listdir(video_path))):
            video_name = os.path.join(sub_folder, video)
            if video_name not in test_videos and video_name.find('mp4') != -1:
                sub_train_videos.append(os.path.join(video_name))
        train_videos.append(sub_train_videos)

    train_txt = open('../save_txt/train_cdf.txt', 'w')
    val_txt = open('../save_txt/val_cdf.txt', 'w')
    for i in range(len(sub_folders)):
        for j in range(len(train_videos[i])):
            if random.random() > 0.2:
                train_txt.write(train_videos[i][j] + '\n')
            else:
                val_txt.write(train_videos[i][j] + '\n')


if __name__ == '__main__':
    # ff_main()
    cdf_main()
