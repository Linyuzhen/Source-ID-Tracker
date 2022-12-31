
import cv2
import math
import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from DeepFakeMask import dfl_full,facehull,components,extended

import os


    

class ff_df_Dataloader(Dataset):
    def __init__(self, root_path="", fake_video_names=[], source_video_names=[], phase='train', test_frame_nums=300,
                 transform=None, size=(256, 256)):
        assert phase in ['train', 'valid', 'test']
        self.root_path = root_path
        self.fake_video_names = []
        self.source_video_names = []
        self.source_video_landmarks = OrderedDict()
        for fake_video_name, source_video_name in tqdm(zip(fake_video_names, source_video_names)):
            aa = source_video_name.split('/')
            json_file = os.path.join(aa[0], aa[1], aa[2], aa[3], 'dlib_landmarks', aa[4] + '.json')
            json_path = os.path.join(self.root_path, json_file)
            # print(json_path)
            if os.path.isfile(json_path):
                all_landmarks = self.load_landmarks(json_path)
                if len(all_landmarks) != 0:
                    self.source_video_landmarks[source_video_name] = all_landmarks
                    self.fake_video_names.append(fake_video_name)
                    self.source_video_names.append(source_video_name)
        self.phase = phase
        self.test_frame_nums = test_frame_nums
        self.transform = transform
        self.size = size
        if phase != 'train':
            self.fake_image_names, self.source_image_names = self.load_image_name()
        else:
            print('The number of test videos is : %d' % len(fake_video_names))
      
    def load_image_name(self):
        fake_image_names = []
        source_image_names = []
        for idx, fake_video_name in tqdm(enumerate(self.fake_video_names)):
            random.seed(2021)
            video_path = os.path.join(self.root_path, fake_video_name)
            source_video_name = self.source_video_names[idx]
            all_frame_names = os.listdir(video_path)
            frame_names = []
            for image_name in all_frame_names:
                if int(image_name.split('/')[-1].replace('.png', '')) % 10 == 0 and \
                                        self.source_video_landmarks[source_video_name].get(image_name) is not None:
                    frame_names.append(image_name)
            if len(frame_names) > self.test_frame_nums:
                frame_names = random.sample(frame_names, self.test_frame_nums)
            for image_name in frame_names:
                fake_image_names.append(os.path.join(fake_video_name, image_name))
                source_image_names.append(os.path.join(source_video_name, image_name))
        return fake_image_names, source_image_names

    def load_landmarks(self, landmarks_file):
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

    def get_label(self, path):
        if path.find('youtube') != -1:
            label = 0
        elif path.find('Deepfakes') != -1:
            label = 1
        elif path.find('FaceSwap') != -1:
            label = 2
        elif path.find('FaceShifter') != -1:
            label = 3
        elif path.find('Face2Face') != -1:
            label = 4
        elif path.find('NeuralTextures') != -1:
            label = 5
        return label


    def read_png(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        # if random.random() >0.75:
            # quality = random.randint(75, 100)
            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            # face_img_encode = cv2.imencode('.jpg', image, encode_param)[1]
            # image = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return image      

        
    def __getitem__(self, index):
        if self.phase == 'train':
            fake_video_name = self.fake_video_names[index]
            source_video_name = self.source_video_names[index]
            all_landmarks = self.source_video_landmarks[source_video_name]
            
            image_name = random.sample(list(all_landmarks.keys()), 1)[0]

            fake_video_path = os.path.join(self.root_path, fake_video_name)
            fake_image_path = os.path.join(fake_video_path, image_name)
            fake_image = self.read_png(fake_image_path)

            source_image_path = os.path.join(self.root_path, source_video_name, image_name)
            source_image = self.read_png(source_image_path)
            
            face_mask = facehull(landmarks=all_landmarks[image_name].astype('int32'),face=cv2.resize(source_image, self.size), channels=3).mask/255.0
            # face_mask = np.expand_dims(face_mask, axis=2)
           
            fake_image = self.transform(image=fake_image)["image"]
            face_data = self.transform(image=source_image, mask=face_mask)    
            source_image = face_data['image']
            face_mask = face_data['mask'].permute(2,0,1)
            
            return fake_image, source_image, face_mask
            
        else:
            fake_image_path = os.path.join(self.root_path, self.fake_image_names[index])
            source_image_path = os.path.join(self.root_path, self.source_image_names[index])
            source_video_name = '/'.join(source_image_path.split('/')[4:-1])
            all_landmarks = self.source_video_landmarks[source_video_name]
            
            fake_image = self.read_png(fake_image_path)
            source_image = self.read_png(source_image_path)
            face_mask = facehull(landmarks=all_landmarks[source_image_path.split('/')[-1]].astype('int32'),face=cv2.resize(source_image, self.size), channels=3).mask/255.0

            fake_image = self.transform(image=fake_image)["image"]
            face_data = self.transform(image=source_image, mask=face_mask)    
            source_image = face_data['image']
            face_mask = face_data['mask'].permute(2,0,1)

            return fake_image, source_image, face_mask

    def __len__(self):
        if self.phase == 'train':  # load Rebalanced image data
            return len(self.fake_video_names)
        else:  # load all image
            return len(self.fake_image_names)
