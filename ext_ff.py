import torch
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image

from facenet_pytorch import MTCNN

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def extract_video(input_dir, model, scale=1.3, smallest_h=0, smallest_w=0, gp=30):
    reader = cv2.VideoCapture(input_dir)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 64
    face_boxes = []
    face_images = []
    face_index = []
    original_frames = OrderedDict()
    halve_frames = OrderedDict()
    index_frames = OrderedDict()
    for i in range(frames_num):
        reader.grab()
        success, frame = reader.retrieve()
        frame_shape = frame.shape
        if smallest_h != frame_shape[0]:
            diff = frame_shape[0] - smallest_h
            m = diff // 2
            frame = frame[m:-m, :, :]
        if smallest_w != frame_shape[1]:
            diff = frame_shape[1] - smallest_w
            m = diff // 2
            frame = frame[:, m:-m, :]

        if i % gp == 0:
            if not success:
                continue
            original_frames[i] = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize(size=[s // 2 for s in frame.size])
            halve_frames[i] = frame
            index_frames[i] = i

    original_frames = list(original_frames.values())
    halve_frames = list(halve_frames.values())
    index_frames = list(index_frames.values())
    print(input_dir[-7:])
    for i in range(0, len(halve_frames), batch_size):
        batch_boxes, batch_probs, batch_points = model.detect(halve_frames[i:i + batch_size], landmarks=True)
        None_array = np.array([], dtype=np.int16)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                pass
            else:
                print("no face in:{}_{}".format(input_dir[-7:],index))
                batch_boxes[index] = batch_boxes[index-1]
                batch_probs[index] = batch_probs[index-1]
                batch_points[index] = batch_points[index-1]
                # batch_probs[index] = [0]
                continue

        batch_boxes, batch_probs, batch_points = model.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    halve_frames[i:i + batch_size],
                                                                    method="probability")
        # method="largest")
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                # crop = original_frames[i:i + batch_size][index][ymin:ymin+size_bb, xmin:xmin+size_bb]
                face_index.append(index_frames[i:i + batch_size][index])
                face_boxes.append([ymin, ymin + size_bb, xmin, xmin + size_bb])
                crop = original_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]
                face_images.append(crop)
            else:
                continue

    return face_images, face_boxes, face_index


def get_smallest_hw(video_root_path, real_sub_path, real_video, deepfake_sub_path, fake_video):
    real_video_path = os.path.join(video_root_path, real_sub_path, real_video)
    reader = cv2.VideoCapture(real_video_path)
    reader.grab()
    success, frame = reader.retrieve()
    frame_real_shape = frame.shape
    del reader

    reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path, fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_deepfakes_shape = frame.shape
    del reader

    reader = cv2.VideoCapture(
        os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_face2face_shape = frame.shape
    del reader

    reader = cv2.VideoCapture(
        os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_NeuralTextures_shape = frame.shape
    del reader

    reader = cv2.VideoCapture(
        os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_FaceSwap_shape = frame.shape
    del reader

    reader = cv2.VideoCapture(
        os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceShifter'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_FaceShifter_shape = frame.shape
    del reader

    smallest_h = min(frame_real_shape[0], frame_deepfakes_shape[0], frame_face2face_shape[0],
                     frame_NeuralTextures_shape[0], frame_FaceSwap_shape[0], frame_FaceShifter_shape[0])
    smallest_w = min(frame_real_shape[1], frame_deepfakes_shape[1], frame_face2face_shape[1],
                     frame_NeuralTextures_shape[1], frame_FaceSwap_shape[1], frame_FaceShifter_shape[1])

    return smallest_h, smallest_w


def main(video_root_path, image_root_path, real_sub_path, deepfake_sub_path, real_videos, fake_videos):
    scale = 1.3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    detector = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)

    for idx in tqdm(range(len(real_videos))):

        real_video_path = os.path.join(video_root_path, real_sub_path, real_videos[idx])
        check_path = os.path.join(image_root_path, real_sub_path, real_videos[idx])

        if os.path.exists(check_path):
            print("{} is existed".format(real_videos[idx]))
            continue

        smallest_h, smallest_w = get_smallest_hw(video_root_path, real_sub_path, real_videos[idx], deepfake_sub_path,
                                                 fake_videos[idx])

        face_images, face_boxes, face_index = extract_video(real_video_path, detector, scale=scale,
                                                            smallest_h=smallest_h, smallest_w=smallest_w)

        temp_save_root_path = os.path.join(image_root_path, real_sub_path, real_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j, index in enumerate(face_index):
            cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % index), face_images[j])

        reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path, fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path, fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]
            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'), fake_videos[idx]))
        f2f_frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'),
                                           fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(f2f_frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]

            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'), fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path,
                                           deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'),
                                           fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]

            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'), fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'),
                                           fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]

            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceShifter'), fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceShifter'),
                                           fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]

            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)


if __name__ == "__main__":
    video_root_path = '/home/linyz/datasets/FaceForensics++'
    image_root_path = '/home/linyz/datasets/FF-face'
    real_sub_path = 'original_sequences/youtube/c40/videos'
    deepfake_sub_path = 'manipulated_sequences/Deepfakes/c40/videos'
    if not os.path.isdir(image_root_path):
        os.mkdir(image_root_path)

    temp_image_root_path = image_root_path
    for name in real_sub_path.split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'Face2Face').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'FaceSwap').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'NeuralTextures').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'FaceShifter').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    f = open('splits/test.json', 'r')
    test_json = json.load(f)
    fake_videos = []
    real_videos = []
    for video_name in test_json:
        fake_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_videos.append(video_name[0] + '.mp4')
        real_videos.append(video_name[1] + '.mp4')

    f = open('splits/val.json', 'r')
    test_json = json.load(f)
    for video_name in test_json:
        fake_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_videos.append(video_name[0] + '.mp4')
        real_videos.append(video_name[1] + '.mp4')

    f = open('splits/train.json', 'r')
    test_json = json.load(f)
    for video_name in test_json:
        fake_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_videos.append(video_name[0] + '.mp4')
        real_videos.append(video_name[1] + '.mp4')

    main(video_root_path, image_root_path, real_sub_path, deepfake_sub_path, real_videos, fake_videos)
