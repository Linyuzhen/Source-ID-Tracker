import cv2
import json
import random
from torch.utils.data import Dataset
import numpy as np
from collections import OrderedDict
from DeepFakeMask import dfl_full,facehull,components,extended
import os



class Test_Dataloader(Dataset):
    def __init__(self, fake_frame_path='xxxx.mp4', source_frame_path='xxx.mp4', phase='valid', test_frame_nums=500,
                 landmarks_file_path='xxx.mp4.json', transform=None, size=(256, 256)):
        assert phase in ['valid', 'test']
        self.fake_frame_path = fake_frame_path
        self.source_frame_path = source_frame_path
        self.landmarks_file_path = landmarks_file_path
        self.all_landmarks = self.load_landmarks(self.landmarks_file_path)

        self.phase = phase
        self.test_frame_nums = test_frame_nums
        self.transform = transform
        self.size = size
        self.fake_image_names, self.source_image_names = self.load_image_name()

    def load_image_name(self):
        fake_image_names = []
        source_image_names = []
        all_frame_names = os.listdir(self.fake_frame_path)
        frame_names = []
        for image_name in all_frame_names:
            if int(image_name.split('/')[-1].replace('.png', '')) % 10 == 0 and \
                                    self.all_landmarks.get(image_name) is not None:
                frame_names.append(image_name)
        if len(frame_names) > self.test_frame_nums:
            random.seed(2021)
            frame_names = random.sample(frame_names, self.test_frame_nums)
        for image_name in frame_names:
            fake_image_names.append(os.path.join(self.fake_frame_path, image_name))
            source_image_names.append(os.path.join(self.source_frame_path, image_name))
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

    def read_png(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return image      

        
    def __getitem__(self, index):
        fake_image_path = self.fake_image_names[index]
        source_image_path = self.source_image_names[index]
        
        fake_image = self.read_png(fake_image_path)
        source_image = self.read_png(source_image_path)
        image_name = source_image_path.split('/')[-1]
        face_mask = facehull(landmarks=self.all_landmarks[image_name].astype('int32'),
                            face=cv2.resize(source_image, self.size), channels=3).mask/255.0

        fake_image = self.transform(image=fake_image)["image"]
        face_data = self.transform(image=source_image, mask=face_mask)    
        source_image = face_data['image']
        face_mask = face_data['mask'].permute(2,0,1)

        return fake_image, source_image, face_mask, image_name

    def __len__(self):
        return len(self.fake_image_names)



class Test_DataloaderV2(Dataset):
    def __init__(self, fake_frame_path='xxxx.mp4', source_frame_path='xxx.mp4', phase='valid', test_frame_nums=500,
                 landmarks_file_path='xxx.mp4.json', transform=None, size=(256, 256)):
        assert phase in ['valid', 'test']
        self.fake_frame_path = fake_frame_path
        self.source_frame_path = source_frame_path
        self.landmarks_file_path = landmarks_file_path
        self.all_landmarks = self.load_landmarks(self.landmarks_file_path)

        self.phase = phase
        self.test_frame_nums = test_frame_nums
        self.transform = transform
        self.size = size
        self.fake_image_names, self.source_image_names = self.load_image_name()

    def load_image_name(self):
        fake_image_names = []
        source_image_names = []
        all_frame_names = os.listdir(self.fake_frame_path)
        frame_names = []
        for image_name in all_frame_names:
            if int(image_name.split('/')[-1].replace('.png', '')) % 10 == 0 and \
                                    self.all_landmarks.get(image_name) is not None:
                frame_names.append(image_name)
        if len(frame_names) > self.test_frame_nums:
            random.seed(2021)
            frame_names = random.sample(frame_names, self.test_frame_nums)
        for image_name in frame_names:
            fake_image_names.append(os.path.join(self.fake_frame_path, image_name))
            source_image_names.append(os.path.join(self.source_frame_path, image_name))
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

    def read_png(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return image      

        
    def __getitem__(self, index):
        fake_image_path = self.fake_image_names[index]
        source_image_path = self.source_image_names[index]
        
        fake_image = self.read_png(fake_image_path)
        source_image = self.read_png(source_image_path)
        image_name = source_image_path.split('/')[-1]
        try:
            landmarks = self.all_landmarks[image_name].astype('int32')
            face_mask = facehull(landmarks=landmarks,
                                face=cv2.resize(source_image, self.size), channels=3).mask/255.0
        except:
            print(source_image_path)
            face_mask = np.ones(cv2.resize(source_image, self.size).shape)
        fake_image = self.transform(image=fake_image)["image"]
        face_data = self.transform(image=source_image, mask=face_mask)    
        source_image = face_data['image']
        face_mask = face_data['mask'].permute(2,0,1)

        return fake_image, source_image, face_mask, image_name

    def __len__(self):
        return len(self.fake_image_names)




if __name__ == '__main__':
    height = 256
    width = 256
    import albumentations as A

    train_transform = A.Compose([ A.Resize(height, width)])

    with open('test.txt', 'r') as f:
        train_videos = f.readlines()
        train_videos = [i.strip() for i in train_videos]

    if not os.path.isdir('try'):
        os.mkdir('try')

    dataset = binary_Rebalanced_Dataloader(root_path='/raid/chenhan/Celeb-DF-v2-face',
                                           video_names=train_videos[0:10], phase='train',
                                           transform=train_transform)
    
    for m in range(len(dataset)):
        image, label = dataset[m]
        print (label)
        # print (image)
        # image = image.numpy().transpose(1, 2, 0)
        print(image.shape)
        cv2.imwrite('try/%d_%d.jpg' % (m, label), image)

        # break
