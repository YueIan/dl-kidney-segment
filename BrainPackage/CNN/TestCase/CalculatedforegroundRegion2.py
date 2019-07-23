from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import PIL
import cv2
# from BagData import test_dataloader, train_dataloader
from BrainPackage.CNN.Dataset.JPGDataForSegment import JPGDataForSegment
from BrainPackage.CNN.Dataset.JPGDataForSegmentEvaluate2 import JPGDataForSegmentEvaluate2
from BrainPackage.CNN.Dataset.RAWDataForSegment import RAWDataForSegment
from BrainPackage.CNN.Model.FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from BrainPackage.CNN.Model.uNet import UNet
from BrainPackage.CNN.Model.PSPNet import PSPNet
from BrainPackage.DataPrepPocessing import RandomDataSelect
from BrainPackage.CNN.Loss.FocalLoss import FocalLoss
from BrainPackage.CNN.Evaluate.KidneyEvaluate import acc1an2
from BrainPackage.CNN.Evaluate.KidneyEvaluate import acc1an2forOneSlicein3D
from BrainPackage.CNN.Evaluate.KidneyEvaluate import acc1an2in3D
from BrainPackage.CNN.Evaluate.KidneyEvaluate import accall
from BrainPackage.CNN.Evaluate.KidneyEvaluate import AccStruct
from BrainPackage.CNN.Evaluate.KidneyEvaluate import SetALLZero
from skimage import transform
# import visdom
class SegmentTestInfo:
    test_result_root =''
    test_mask_root =''
    original_image_root=''
    original_label_root=''
    stamp = ''
    saved_image_root =''
    is_not_test = True
class CalculatedforegroundRegion2:
    '''This is CNN test case  '''
    def __init__(self, test_info):
        self.test_result_root = test_info.test_result_root
        self.test_mask_root = test_info.test_mask_root
        self.original_image_root = test_info.original_image_root
        self.original_label_root = test_info.original_label_root
        self.saved_image_root = test_info.saved_image_root
        self.stamp = test_info.stamp
        self.is_not_test = test_info.is_not_test
    def PerformTestCase(self):
        self.PrepareTestCase(show_vgg_params=False)
        self.RunTestCase()
    def PrepareTestCase(self,show_vgg_params=False):
        #Prepare training data
        self.test_data = JPGDataForSegmentEvaluate2(self.test_result_root, self.test_mask_root, True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, num_workers=0)
        self.evaluate_data = JPGDataForSegmentEvaluate2(self.original_image_root, self.original_label_root, False)
        self.evaluate_loader = torch.utils.data.DataLoader(dataset=self.evaluate_data, batch_size=1, shuffle=False, num_workers=0)
        self.id_list = self.test_data.get_id_list()

    def RunTestCase(self):    

        if not os.path.exists(self.saved_image_root):
            os.mkdir(self.saved_image_root)
        if not os.path.exists(os.path.join(self.saved_image_root, r'limage')):
            os.mkdir(os.path.join(self.saved_image_root, r'limage'))
        if not os.path.exists(os.path.join(self.saved_image_root, r'rimage')):
            os.mkdir(os.path.join(self.saved_image_root, r'rimage'))
        if not os.path.exists(os.path.join(self.saved_image_root, r'llabel')):
            os.mkdir(os.path.join(self.saved_image_root, r'llabel'))
        if not os.path.exists(os.path.join(self.saved_image_root, r'rlabel')):
            os.mkdir(os.path.join(self.saved_image_root, r'rlabel'))
        acc_index =0;
        patients =list(self.id_list.keys())
        patients = sorted(patients);
        current_patient_index =0;

        acc_result = AccStruct()
        for index, (bag_msk_np, output_np) in enumerate(self.test_loader):
            acc1an2forOneSlicein3D(bag_msk_np.numpy(),output_np.numpy(),acc_result, index)
            if index - acc_index == self.id_list[patients[current_patient_index]]-1:
                acc_tum,acc_kidney,acc_fore = accall(acc_result,acc_index, 0.4, True)
                acc_index = acc_index + self.id_list[patients[current_patient_index]]
                current_patient_index = current_patient_index +1;


        fpl = open(self.saved_image_root+ "\\information3.csv" , 'w')
        for index in range(len(acc_result.lbbox)):
            image_width = 512
            image_height = 512

            if acc_result.is_valid[index]:
                left1 = (acc_result.lbbox[index].centerx - 36) * 8 // 3
                top1 = (acc_result.lbbox[index].centery - 36) * 8 // 3
                left2 = (acc_result.rbbox[index].centerx - 36) * 8 // 3
                top2 = (acc_result.rbbox[index].centery - 36) * 8 // 3
                fpl.write(patients[index] + ',1,'+ str(self.id_list[patients[index]]) + ','+
                str(image_width) + ',' +str(image_height) +',' + str(left1) + ',' + str(top1) + ','+ str(left2)  + ',' + str(top2)+'\n')
            else:
                fpl.write(patients[index] + ',0,'+ str(self.id_list[patients[index]]) + ','+
                str(image_width) + ',' +str(image_height) +'\n')
        fpl.close()

        current_patient_index = 0
        acc_index = 0
        left1 = (acc_result.lbbox[current_patient_index].centerx - 36) * 8 // 3
        # left1 = (acc_result.lbbox[test_result_index].centerx - 96*48//199) * 199 // 48
        right1 = left1 + 192
        top1 = (acc_result.lbbox[current_patient_index].centery - 36) * 8 // 3
        bottom1 = top1 + 192
        # left2 = (acc_result.rbbox[test_result_index].centerx - 96*48//199) * 199 // 48
        left2 = (acc_result.rbbox[current_patient_index].centerx - 36) * 8 // 3
        right2 = left2 + 192
        top2 = (acc_result.rbbox[current_patient_index].centery - 36) * 8 // 3
        bottom2 = top2 + 192
        print(patients[current_patient_index])


        for index, (raw_label, raw_image) in enumerate(self.evaluate_loader):
            if acc_result.valid_slice[current_patient_index][index -acc_index] > 0 and acc_result.is_valid[current_patient_index]: 
                raw_image = raw_image.numpy()
                if raw_image.shape[1] !=512 or raw_image.shape[2] != 512:
                    print("zhatian!")
                raw_image = np.reshape(raw_image, [raw_image.shape[1], raw_image.shape[2],raw_image.shape[3]])   
                raw_image1 = np.zeros((192,192,3))
                raw_image1 = raw_image[top1:bottom1,left1:right1,:]
                raw_image2 = np.zeros((192,192,3))
                raw_image2 = raw_image[top2:bottom2, left2:right2,:]

                raw_image1 =  PIL.Image.fromarray(raw_image1) 
                raw_image1.save(open('{:s}\\limage\\left_small_patch_{:s}_{:05d}.png'.format(self.saved_image_root, patients[current_patient_index], index-acc_index), 'wb'))
                raw_image2 =  PIL.Image.fromarray(raw_image2) 
                raw_image2.save(open('{:s}\\rimage\\right_small_patch_{:s}_{:05d}.png'.format(self.saved_image_root,patients[current_patient_index], index-acc_index), 'wb'))
                if self.is_not_test:
                    raw_label = raw_label.numpy()
                    raw_label = np.reshape(raw_label, [raw_label.shape[1], raw_label.shape[2],raw_label.shape[3]])   
                    raw_label1 = np.zeros((192,192,3))
                    raw_label1 = raw_label[top1:bottom1,left1:right1,:]
                    raw_label2 = np.zeros((192,192,3))
                    raw_label2 = raw_label[top2:bottom2, left2:right2,:]
                    raw_label1 =  PIL.Image.fromarray(raw_label1) 
                    r,g,b = raw_label1.split()
                    raw_label1 =  PIL.Image.merge("RGB", (b, g, r))
                    raw_label1.save(open('{:s}\\llabel\\left_small_patch_{:s}_{:05d}.png'.format(self.saved_image_root,patients[current_patient_index], index-acc_index), 'wb'))
                    raw_label2 =  PIL.Image.fromarray(raw_label2) 
                    r,g,b = raw_label2.split()
                    raw_label2 =  PIL.Image.merge("RGB", (b, g, r))
                    raw_label2.save(open('{:s}\\rlabel\\right_small_patch_{:s}_{:05d}.png'.format(self.saved_image_root,patients[current_patient_index], index-acc_index), 'wb'))


            if index - acc_index == self.id_list[patients[current_patient_index]]-1:
                acc_index = acc_index + self.id_list[patients[current_patient_index]]
                current_patient_index = current_patient_index +1;
                if current_patient_index <=len(patients)-1:
                    print(patients[current_patient_index])
                    left1 = (acc_result.lbbox[current_patient_index].centerx - 36) * 8 // 3
                    right1 = left1 + 192
                    top1 = (acc_result.lbbox[current_patient_index].centery - 36) * 8 // 3
                    bottom1 = top1 + 192
                    left2 = (acc_result.rbbox[current_patient_index].centerx - 36) * 8 // 3
                    right2 = left2 + 192
                    top2 = (acc_result.rbbox[current_patient_index].centery - 36) * 8 // 3
                    bottom2 = top2 + 192

        SetALLZero(acc_result)

if __name__ == "__main__":

    train(epo_num=100, show_vgg_params=False)
