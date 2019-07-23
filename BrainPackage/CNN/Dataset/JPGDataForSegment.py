import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import cv2
from BrainPackage.CNN.Utility.onehot import onehot
from BrainPackage.Exception import ProgramException

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform = transforms.Compose([
    transforms.ToTensor()
])
class JPGDataForSegment(Dataset):

    def __init__(self, image_path, mask_path, transform=None, data_size =160, segment_number = 2, weight = [0,0,1,1], is_need_onehot = True):
        self.transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.image = []
        self.mask = []
        if not os.path.exists(image_path):
            raise ProgramException('JPGDataForSegment','The following sub folder is not exist:\n' + image_path)
        try:
            file_list = sorted(os.listdir(image_path))
        except:
            raise ProgramException('JPGDataForSegment','The following folder has unexpected sub folder:\n' + image_path)
        input_image_number = len(file_list)
        if input_image_number <= 0:
            raise ProgramException('JPGDataForSegment', 'The following sub folder has no file:\n' + image_path)
        for file in file_list:
            image_full_path = os.path.join(image_path, file)
            mask_full_path = os.path.join(mask_path, file) # same name
            if not os.path.exists(image_full_path):
                raise ProgramException('JPGDataForSegment', 'The following file is not exist:\n' + image_full_path)
            if not os.path.exists(mask_full_path):
                raise ProgramException('JPGDataForSegment', 'The following file is not exist:\n' + mask_full_path)
            self.image.append(image_full_path)
            self.mask.append(mask_full_path)
        self.image_path = image_path
        self.mask_path = mask_path
        self.data_size = data_size
        self.segment_number = segment_number
        self.weight = weight
        self.is_need_onehot = is_need_onehot
    def __len__(self):
        return len(self.image)
    def sortresult(self):
        self.image.sort()
        self.mask.sort()
        result_number = []
        lastdata = self.image[0].split('_')[-2]
        for i in range(len(self.image) - 1):
            currentdata = self.image[i+1].split('_')[-2]
            if currentdata != lastdata:
                lastdata = currentdata
                result_number.append(i)
        result_number.append(len(self.image) - 1)       
        return result_number
            
    def __getitem__(self, idx):
        img_name = self.image[idx]
        mask_name = self.mask[idx]
        name = img_name.split('\\')[-1]
        imgA = cv2.imread(img_name)
        imgA = cv2.resize(imgA, (self.data_size, self.data_size))
        raw_image = imgA
        imgB = cv2.imread(mask_name, 1)
        imgB = cv2.resize(imgB, (self.data_size, self.data_size))
        if imgB.shape[2] == 1:
            imgB = imgB/255
        elif imgB.shape[2] == 3:
            newimgB = np.zeros((imgB.shape[0],imgB.shape[1]))
            imgB = imgB.astype('float64')
            newimgB[:,:] = self.weight[0] * imgB[:,:,0] + self.weight[1] * imgB[:,:,1] + self.weight [2] * imgB[:,:,2]
            # because mask is read
            imgB = newimgB/255 + self.weight[3] #[w,h]
        if self.is_need_onehot == True:
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, self.segment_number)
            imgB = imgB.transpose(2,0,1) #[c,w,h]
            imgB = torch.FloatTensor(imgB)
        else:
            #  ONLY REMAIN KIDNEY AND TUMOR
            imgc = np.zeros_like(imgA)
            imgc[np.greater(imgB,0)]= 1
            # imgA = imgc*imgA;
            imgB = torch.LongTensor(imgB)

        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)    
        return imgA, imgB, raw_image, name
    
    def loadDataFromExistedFileList(
        self,
        data
    ):
        self.image = []
        self.mask = []
        for index in range(len(data)):
            self.image.append(os.path.join(self.image_path, data[index]))
            self.mask.append(os.path.join(self.mask_path, data[index])) # same name

    def SaveDataAndLabelToTxt(
        self,
        fileName
    ):
        fp = open(fileName, mode='w')
        for index in range(len(self.image)):
            fp.write(self.image[index])
            fp.write(',')
            fp.write(str(self.mask[index]))
            fp.write('\n')
        fp.close()
    def LoadDataAndLabelFromtxt(
        self,
        fileName
    ):
        self.image = []
        self.mask = []
        fp = open(fileName, mode='r')
        while True:
            line = fp.readline()
            if len(line) == 0:
                break
            else:
                info = line.strip('\n')
                info = info.split(',')
                self.image.append(info[0])
                self.mask.append(info[1])
        fp.close()
# bag = BagDataset(transform)

# train_size = int(0.9 * len(bag))
# test_size = len(bag) - train_size
# train_dataset, test_dataset = random_split(bag, [train_size, test_size])

# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


# if __name__ =='__main__':

#     for train_batch in train_dataloader:
#         print(train_batch)

#     for test_batch in test_dataloader:
#         print(test_batch)
