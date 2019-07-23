import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import cv2
from BrainPackage.CNN.Utility.onehot import onehot
from BrainPackage.Exception import ProgramException
import re
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform = transforms.Compose([
    transforms.ToTensor()
])
class JPGDataForSegmentEvaluate2(Dataset):

    def __init__(self, image_path, mask_path, is_transform):
        # self.transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.image = []
        self.mask = []
 
        file_list = sorted(os.listdir(image_path))
        input_image_number = len(file_list)
        current_data =''
        first_image = True
        self.id_list ={}
        for file in file_list:
            info = re.split(" |_|\.",file)
            image_full_path = os.path.join(image_path, file) 
            mask_full_path = os.path.join(mask_path, file) # same name
            self.image.append(image_full_path)
            self.mask.append(mask_full_path)
            if first_image == True:
                temp_slice = 1
                temp_image_id = info[-3]
                first_image = False
                continue
            if temp_image_id == info[-3]:
                temp_slice = temp_slice + 1
            else:
                self.id_list[temp_image_id] = temp_slice
                temp_slice = 1
                temp_image_id = info[-3]

        self.id_list[temp_image_id] = temp_slice
        # self.id_list = sorted(self.id_list.items(), key=lambda d: d[0])
        # self.image.sort()
        # self.mask.sort()
        self.image_path = image_path
        self.mask_path = mask_path
        self.is_transform = is_transform
    def __len__(self):
        return len(self.image)
    def get_id_list(self):
        return self.id_list       
    def __getitem__(self, idx):
        img_name = self.image[idx]
        mask_name = self.mask[idx]
        if self.is_transform:
            imgA = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            imgB = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            t_imagea = np.zeros_like(imgA)
            t_imageb = np.zeros_like(imgB)
            t_imagea[np.equal(imgA[:,:],127)] = 1
            t_imagea[np.equal(imgA[:,:],255)] = 2
            t_imageb[np.equal(imgB[:,:],127)] = 1
            t_imageb[np.equal(imgB[:,:],255)] = 2
            # result
            t_imagea = torch.LongTensor(t_imagea)
            # label
            t_imageb = torch.LongTensor(t_imageb)  
            return t_imageb, t_imagea
        else:
            # image
            imgA = cv2.imread(img_name, 1)
            # label
            imgB = cv2.imread(mask_name, 1)
            t_imagea = imgA
            t_imageb = imgB
            return t_imageb, t_imagea

       