from starter_code.utils import load_case
from starter_code.visualize import visualize
from starter_code.evaluation import evaluate
import os
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib



# rough_segment_result_path = r'C:\kidneydata_inte\rough_segmentation_data\result2'
# detail_segment_result_path =r'C:\kidneydata_inte\rough_segmentation_data\result4'
# detail_information_file =r'C:\kidneydata_inte\rough_segmentation_data\output\information.csv'

def output_result_data_nii(rough_segment_result_path,detail_segment_result_path, 
    detail_information_file, detail_segment, resultnii_output_path):
    if not os.path.exists(resultnii_output_path):
        os.mkdir(resultnii_output_path)
    fp = open(detail_information_file, mode='r')
    while True:
        line = fp.readline()
        if len(line) == 0:
            break
        else:
            info = line.strip('\n')
            info = info.split(',')
            output_label_name ='{:s}\\prediction_{:s}.raw'.format(resultnii_output_path,info[0])
            output_label_name2 ='{:s}\\prediction_{:s}.nii.gz'.format(resultnii_output_path,info[0])
            # output_label_name3 ='t_prediction_{:s}.raw'.format(info[0])
            output_result = np.zeros([int(info[2]),int(info[3]),int(info[4])],dtype=np.uint8)
            if int(info[1]) == 1: 
                # detail segment
                for i in range(int(info[2])):
                    filename1 = "left_small_patch_{}_{:05d}.png".format(info[0],i)
                    filename2 = "right_small_patch_{}_{:05d}.png".format(info[0],i)
                    lresult_filename = detail_segment_result_path + "\\left\\" + filename1
                    rresult_filename = detail_segment_result_path + "\\right\\" + filename2
                    if Path(lresult_filename).exists():
                        imgA = cv2.imread(lresult_filename, cv2.IMREAD_GRAYSCALE)
                        t_imagea = np.zeros_like(imgA, dtype = np.uint8)
                        t_imagea[np.equal(imgA[:,:],127)] = 1
                        t_imagea[np.equal(imgA[:,:],255)] = 2
                        left1 = int(info[5])
                        top1 = int(info[6])
                        right1 = left1 + detail_segment;
                        bottom1 = top1 + detail_segment;
                        output_result[i,top1:bottom1,left1:right1] = t_imagea; 
                    if Path(rresult_filename).exists():
                        imgA = cv2.imread(rresult_filename, cv2.IMREAD_GRAYSCALE)
                        t_imagea = np.zeros_like(imgA, dtype = np.uint8)
                        t_imagea[np.equal(imgA[:,:],127)] = 1
                        t_imagea[np.equal(imgA[:,:],255)] = 2
                        left1 = int(info[7])
                        top1 = int(info[8])
                        right1 = left1 + detail_segment;
                        bottom1 = top1 + detail_segment;
                        output_result[i,top1:bottom1,left1:right1] = t_imagea; 
                output_result.tofile(output_label_name)
                img = nib.Nifti1Image(output_result,np.eye(4))
                nib.save(img,output_label_name2)
            else:
                # rough segment
                for i in range(int(info[2])):
                    filename = "case_{}_{:05d}.png".format(info[0],i)
                    filepath = rough_segment_result_path + "\\" + filename
                    imgA = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    imgA = cv2.resize(imgA, (int(info[3]), int(info[4])),interpolation=cv2.INTER_NEAREST)
                    t_imagea = np.zeros_like(imgA)
                    t_imagea[np.equal(imgA[:,:],127)] = 1
                    t_imagea[np.equal(imgA[:,:],255)] = 2
                    output_result[i,:,:] = t_imagea; 
                output_result.tofile(output_label_name)
                img = nib.Nifti1Image(output_result,np.eye(4))
                nib.save(img,output_label_name2)
                vol = nib.load(output_label_name2)
                vol = vol.get_data()
    fp.close()


def output_evaluation_data_nii_and_get_score(input_path, rough_segment_result_path, detail_segment_result_path, 
    detail_information_file, detail_segment, resultnii_output_path):

    if not os.path.exists(resultnii_output_path):
        os.mkdir(resultnii_output_path)
    a1=[]
    b1=[]
    info_array=[]
    data_number=0;
    fp = open(detail_information_file, mode='r')
    while True:
        line = fp.readline()
        if len(line) == 0:
            break
        else:
            info = line.strip('\n')
            info = info.split(',')
            output_label_name ='{:s}\\prediction_{:s}.raw'.format(resultnii_output_path,info[0])
            output_label_name2 ='{:s}\\prediction_{:s}.nii.gz'.format(resultnii_output_path,info[0])
            output_result = np.zeros([int(info[2]),int(info[3]),int(info[4])],dtype=np.uint8)
            if int(info[1]) == 1: 
                # detail segment
                for i in range(int(info[2])):
                    filename1 = "left_small_patch_{}_{:05d}.png".format(info[0],i)
                    filename2 = "right_small_patch_{}_{:05d}.png".format(info[0],i)
                    lresult_filename = detail_segment_result_path + "\\left\\" + filename1
                    rresult_filename = detail_segment_result_path + "\\right\\" + filename2
                    if Path(lresult_filename).exists():
                        imgA = cv2.imread(lresult_filename, cv2.IMREAD_GRAYSCALE)
                        t_imagea = np.zeros_like(imgA, dtype = np.uint8)
                        t_imagea[np.equal(imgA[:,:],127)] = 1
                        t_imagea[np.equal(imgA[:,:],255)] = 2
                        # imgA = cv2.imread(lresult_filename)
                        # t_imagea = np.zeros([192,192], dtype = np.uint8)
                        # t_imagea[np.equal(imgA[:,:,2],255)] = 1
                        # t_imagea[np.equal(imgA[:,:,0],255)] = 2
                        left1 = int(info[5])
                        top1 = int(info[6])
                        right1 = left1 + detail_segment;
                        bottom1 = top1 + detail_segment;
                        output_result[i,top1:bottom1,left1:right1] = t_imagea; 
                    if Path(rresult_filename).exists():
                        imgA = cv2.imread(rresult_filename, cv2.IMREAD_GRAYSCALE)
                        t_imagea = np.zeros_like(imgA, dtype = np.uint8)
                        t_imagea[np.equal(imgA[:,:],127)] = 1
                        t_imagea[np.equal(imgA[:,:],255)] = 2
                        # imgA = cv2.imread(rresult_filename)
                        # t_imagea = np.zeros([192,192], dtype = np.uint8)
                        # t_imagea[np.equal(imgA[:,:,2],255)] = 1
                        # t_imagea[np.equal(imgA[:,:,0],255)] = 2
                        left1 = int(info[7])
                        top1 = int(info[8])
                        right1 = left1 + detail_segment;
                        bottom1 = top1 + detail_segment;
                        output_result[i,top1:bottom1,left1:right1] = t_imagea; 
                output_result.tofile(output_label_name)
                img = nib.Nifti1Image(output_result,np.eye(4))
                nib.save(img,output_label_name2)
                [a,b] = evaluate(input_path, int(info[0]),img)
                print(info[0],a,b)
                a1.append(a)
                b1.append(b)
                info_array.append(info[0])
            else:
                # rough segment
                for i in range(int(info[2])):
                    filename = "case_{}_{:05d}.png".format(info[0],i)
                    filepath = rough_segment_result_path + "\\" + filename
                    imgA = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    imgA = cv2.resize(imgA, (int(info[3]), int(info[4])),interpolation=cv2.INTER_NEAREST)
                    t_imagea = np.zeros_like(imgA)
                    t_imagea[np.equal(imgA[:,:],127)] = 1
                    t_imagea[np.equal(imgA[:,:],255)] = 2
                    output_result[i,:,:] = t_imagea; 
                output_result.tofile(output_label_name)
                img = nib.Nifti1Image(output_result,np.eye(4))
                nib.save(img,output_label_name2)

                [a,b] = evaluate(input_path, int(info[0]),img)
                print(info[0],a,b)            
                a1.append(a)
                b1.append(b)
                info_array.append(info[0])
    fp.close()
    fpl = open(resultnii_output_path+ "\\result.csv" , 'w')
    for i in range(len(a1)):
        fpl.write(info_array[i] + ',' + str(a1[i]) +','+str(b1[i]) +'\n')
    fpl.write('Ave,' + str(sum(a1)/len(a1)) +','+str(sum(b1)/len(b1)) +'\n')
    fpl.close()
