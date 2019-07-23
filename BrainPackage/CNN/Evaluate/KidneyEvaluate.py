import numpy as np
from scipy.ndimage import measurements, morphology
import PIL


class Boundbox:
    centerx = 0
    centery = 0
    width = 0
    height = 0


class AccStruct:
    acc1_1_tt = 0
    acc1_1_tf = 0
    acc1_1_ft = 0
    acc2_1_tt = 0
    acc2_1_tf = 0
    acc2_1_ft = 0
    acc_2_tt = 0
    acc_2_tf = 0
    acc_2_ft = 0
    center_index = -1
    center_indexk = -1
    foreground_number = [] 
    labelforeground_number = []
    center_foreground_number = 0
    center_kidney_number = 0
    center_image = np.zeros((1, 1))
    center_kidney_image = np.zeros((1,1))
    # need not clear when process one data, it used for datasets
    num_of_sum = 0
    volume_index = 0
    lbbox = []
    rbbox = []
    valid_slice = []
    is_valid = []
    label_point_number =[]
    key_slice =[]


def SetZero(accresult):
    accresult.acc1_1_tt = 0
    accresult.acc1_1_tf = 0
    accresult.acc1_1_ft = 0
    accresult.acc2_1_tt = 0
    accresult.acc2_1_tf = 0
    accresult.acc2_1_ft = 0
    accresult.acc_2_tt = 0
    accresult.acc_2_tf = 0
    accresult.acc_2_ft = 0 
    accresult.center_index = -1
    accresult.center_foreground_number = 0
    accresult.foreground_number = []
    accresult.labelforeground_number = []
    accresult.center_image = np.zeros((1, 1))

    accresult.center_kidney_number = 0
    accresult.center_kidney_image = np.zeros((1, 1))

def SetALLZero(accresult):
    accresult.acc1_1_tt = 0
    accresult.acc1_1_tf = 0
    accresult.acc1_1_ft = 0
    accresult.acc2_1_tt = 0
    accresult.acc2_1_tf = 0
    accresult.acc2_1_ft = 0
    accresult.acc_2_tt = 0
    accresult.acc_2_tf = 0
    accresult.acc_2_ft = 0
    accresult.center_index = -1
    accresult.center_indexk = -1
    accresult.foreground_number = [] 
    accresult.labelforeground_number = []
    accresult.center_foreground_number = 0
    accresult.center_kidney_number = 0
    accresult.center_image = np.zeros((1, 1))
    accresult.center_kidney_image = np.zeros((1,1))
    # need not clear when process one data, it used for datasets
    accresult.num_of_sum = 0
    accresult.volume_index = 0
    accresult.lbbox = []
    accresult.rbbox = []
    accresult.valid_slice = []
    accresult.is_valid = []
    accresult.label_point_number =[]
    accresult.key_slice =[]
def acc1an2forOneSlicein3D(label, result, accresult, index):
    shp = label.shape
    label_2 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_1 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_0 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_01 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_02 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_12 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_2 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_1 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_0 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_01 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_02 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_12 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    allone = np.ones((shp[1], shp[2]), dtype=np.float32)

    label_0[np.equal(label[0, :, :], 0)] = 1
    label_1[np.equal(label[0, :, :], 1)] = 1
    label_2[np.equal(label[0, :, :], 2)] = 1
    label_12 = allone - label_0
    label_01 = allone - label_2
    label_02 = allone - label_1

    result_0[np.equal(result[0, :, :], 0)] = 1
    result_1[np.equal(result[0, :, :], 1)] = 1
    result_2[np.equal(result[0, :, :], 2)] = 1
    result_12 = allone - result_0
    result_01 = allone - result_2
    result_02 = allone - result_1

    # tumor(2) will be defined as background for kidney segmentation
    acc1_1_tt = np.sum(label_1 * result_1)  
    acc1_1_tf = np.sum(label_1 * result_02)
    acc1_1_ft = np.sum(label_02 * result_1)
    # tumor(2) will be defined as kidney for kidney segmentation
    acc2_1_tt = np.sum(label_12 * result_12)
    acc2_1_tf = np.sum(label_12 * result_0)
    acc2_1_ft = np.sum(label_0 * result_12)
    #only tumor will be defined as tumor for tumor segmentation
    acc_2_tt = np.sum(label_2 * result_2)
    acc_2_tf = np.sum(label_2 * result_01)
    acc_2_ft = np.sum(label_01 * result_2)
    # fore ground point number 
    result12sum = np.sum(result_12)
    result1sum = np.sum(result_1)
    label12sum = np.sum(label_12)
    accresult.acc1_1_tt += acc1_1_tt
    accresult.acc1_1_tf += acc1_1_tf
    accresult.acc1_1_ft += acc1_1_ft
    accresult.acc2_1_tt += acc2_1_tt
    accresult.acc2_1_tf += acc2_1_tf
    accresult.acc2_1_ft += acc2_1_ft
    accresult.acc_2_tt += acc_2_tt
    accresult.acc_2_tf += acc_2_tf
    accresult.acc_2_ft += acc_2_ft
    if result12sum > accresult.center_foreground_number:
        accresult.center_foreground_number = result12sum
        accresult.center_index = index
        accresult.center_image = result_12
        # accresult.center_image.dtype = np.uint8;
    if result1sum > accresult.center_kidney_number:
        accresult.center_kidney_image = result_1;
        accresult.center_kidney_number = result1sum;

    accresult.foreground_number.append(result12sum)
    accresult.labelforeground_number.append(label12sum)


def acc1an2in3D(accresult):
    sum1_1 = accresult.acc1_1_tt * 2 + accresult.acc1_1_tf + accresult.acc1_1_ft 
    sum2_1 = accresult.acc2_1_tt * 2 + accresult.acc2_1_tf + accresult.acc2_1_ft 
    sum_2 = accresult.acc_2_tt * 2 + accresult.acc_2_tf + accresult.acc_2_ft
    # don't divide 0 
    if sum1_1 < 0.01:
        acc1_1 = 0.5
    else:
        acc1_1 = accresult.acc1_1_tt / sum1_1
    
    acc2_1 =0
    if sum2_1 < 0.01:
        acc2_1 = 0.5
    else:
        acc2_1 = accresult.acc2_1_tt / sum2_1

    acc_2 =0  
    if sum_2 < 0.01:
        acc_2 = 0.5
    else:
        acc_2 = accresult.acc_2_tt / sum_2

    acc1 = acc1_1 + acc_2
    acc2 = acc2_1 + acc_2
    SetZero(accresult)       
    return acc1, acc2, acc_2*2

def accall(accresult, base_index =0, slice_thickness =5.0,  more_evaluate = False ):
    sum1_1 = accresult.acc1_1_tt * 2 + accresult.acc1_1_tf + accresult.acc1_1_ft 
    sum2_1 = accresult.acc2_1_tt * 2 + accresult.acc2_1_tf + accresult.acc2_1_ft 
    sum_2 = accresult.acc_2_tt * 2 + accresult.acc_2_tf + accresult.acc_2_ft
    # don't divide 0 
    if sum1_1 < 0.01:
        acc1_1 = 0.5
    else:
        acc1_1 = accresult.acc1_1_tt / sum1_1
    
    acc2_1 =0
    if sum2_1 < 0.01:
        acc2_1 = 0.5
    else:
        acc2_1 = accresult.acc2_1_tt / sum2_1

    acc_2 =0  
    if sum_2 < 0.01:
        acc_2 = 0.5
    else:
        acc_2 = accresult.acc_2_tt / sum_2

    acc1 = acc1_1 + acc_2
    acc2 = acc2_1 + acc_2
    if more_evaluate == True:
        offsetslice = int(10.0 // slice_thickness)
        small_index = 0
        big_index = len(accresult.foreground_number) - 1
        for i in range (accresult.center_index - base_index, -1, -1):
            if accresult.foreground_number[i] == 0 and accresult.foreground_number[max([i-1, 0])]==0 and accresult.foreground_number[max([i-2, 0])]==0:
                small_index = max([i-offsetslice, 0])
                break
        for i in range (accresult.center_index - base_index, len(accresult.foreground_number)):
            if accresult.foreground_number[i] == 0 and accresult.foreground_number[min([i+1, len(accresult.foreground_number) - 1])] == 0 and accresult.foreground_number[min([i+2, len(accresult.foreground_number) -1])]==0:
                big_index = min([i+offsetslice, len(accresult.foreground_number) - 1])
                break
        resultzeroslice = np.ones(len(accresult.foreground_number))
        resultzeroslice[small_index:big_index+1] =0;
        labelslice = np.ones(len(accresult.foreground_number))
        labelpoint = np.array(accresult.labelforeground_number)
        labelslice[np.equal(labelpoint,0)] = 0
        incorrect_slice_number = np.sum(labelslice*resultzeroslice)
        resultoneslice = np.ones(len(accresult.foreground_number))
        resultoneslice[np.equal(resultzeroslice,1)] = 0
        accresult.label_point_number.append(labelpoint)
        key_slice = np.zeros(len(accresult.foreground_number))
        for i in range(16):
            index = (big_index - small_index) * i // 16 + small_index
            key_slice[index] = 1
        accresult.key_slice.append(key_slice)
        accresult.valid_slice.append(labelslice)
        # in fact the the valid slice is follows:
        # accresult.valid_slice.append(resultoneslice)
        # print('invalid slice = %d' % incorrect_slice_number)
        if incorrect_slice_number >0 :
            accresult.num_of_sum += incorrect_slice_number
        im = morphology.binary_opening(accresult.center_image, np.ones((3,3)), iterations=3)
        im2 = morphology.binary_closing(im, np.ones((3,3)), iterations=4)
        # 4
        labels, nbr_objects = measurements.label(im2)
        label_1 = np.zeros_like(labels);
        label_1[np.equal(labels,1)] = 1
        label_2 = np.zeros_like(labels);
        label_2[np.equal(labels,2)] = 1
        # if nbr_objects != 2:
        #     accresult.is_valid.append(False)
        #     label_2 = label_1
        # elif nbr_objects == 2:
        #     accresult.is_valid.append(True)
        if nbr_objects < 2:
            im = morphology.binary_opening(accresult.center_kidney_image, np.ones((3,3)), iterations=3)
            im2 = morphology.binary_closing(im, np.ones((3,3)), iterations=4)
            labels2, nbr_objects2 = measurements.label(im2)
            if nbr_objects2 == 2:
                label_1[np.equal(labels2,1)] = 1
                label_2[np.equal(labels2,2)] = 1
                accresult.is_valid.append(True)
                nbr_objects = nbr_objects2
                # bag_msk_np = PIL.Image.fromarray(labels2)
                # bag_msk_np.save(open('fff{}.png'.format(accresult.volume_index), 'wb'))
            else:
                accresult.is_valid.append(False)
                label_2 = label_1
                # bag_msk_np = PIL.Image.fromarray(labels)
                # bag_msk_np.save(open('ttt{}.png'.format(accresult.volume_index), 'wb'))
        elif nbr_objects == 2:
            accresult.is_valid.append(True)
            # bag_msk_np = PIL.Image.fromarray(labels)
            # bag_msk_np.save(open('fff{}.png'.format(accresult.volume_index), 'wb'))
        else:
            num_label_dict = []
            for i in range(nbr_objects):
                num_label_i = np.sum(np.equal(labels,i+1))
                num_label_dict.append((num_label_i,i))
            result = sorted(num_label_dict, key=lambda x: x[0])  
            accresult.is_valid.append(True)
            label_1[np.equal(labels,result[0][1])] = 1 
            label_2[np.equal(labels,result[1][1])] = 1 
            # bag_msk_np = PIL.Image.fromarray(labels)
            # bag_msk_np.save(open('fff{}.png'.format(accresult.volume_index), 'wb'))
        y, x = label_1.nonzero()
        bbox1 = Boundbox()
        bbox2 = Boundbox()
        bbox1.centerx = (max(x) + min(x))//2
        bbox1.centery = (max(y) + min(y))//2
        bbox1.width = max(x) - min(x)
        bbox1.height = max(y) - min(y)
        y, x = label_2.nonzero()
        bbox2.centerx = (max(x) + min(x))//2
        bbox2.centery = (max(y) + min(y))//2
        bbox2.width = max(x) - min(x)
        bbox2.height = max(y) - min(y)
        if bbox1.centerx < bbox2.centerx:
            accresult.lbbox.append(bbox1)
            accresult.rbbox.append(bbox2)
        else:
            accresult.lbbox.append(bbox2)
            accresult.rbbox.append(bbox1)
        labels = np.uint8(labels*(255//nbr_objects))
        # bag_msk_np = PIL.Image.fromarray(labels)
        # if nbr_objects != 2:
        #     bag_msk_np.save(open('ttt{}.png'.format(accresult.volume_index), 'wb'))
        #     labels2 = np.zeros_like(labels)
        #     labels2[im] =255
        #     bag_msk_np = PIL.Image.fromarray(labels2)
        #     bag_msk_np.save(open('zzz{}.png'.format(accresult.volume_index), 'wb'))
        # else :
        #     bag_msk_np.save(open('fff{}.png'.format(accresult.volume_index), 'wb'))
        print('object number = %d' % nbr_objects)
    SetZero(accresult)       
    accresult.volume_index += 1
    return acc_2*2, acc1_1*2,acc2_1*2

# both kidnry and tumor will be set as 1 
def acctumorandkidney(accresult):
    sum1_1 = accresult.acc1_1_tt * 2 + accresult.acc1_1_tf + accresult.acc1_1_ft
    if sum1_1 < 0.01:
        acc1_1 = 1
    else:
        acc1_1 = accresult.acc1_1_tt * 2  / sum1_1
    return  acc1_1

def acc1an2(label,result):
    shp = label.shape
    label_2 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_1 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_0 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_01 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_02 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    label_12 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_2 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_1 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_0 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_01 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_02 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    result_12 = np.zeros((shp[1], shp[2]), dtype=np.float32)
    allone = np.ones((shp[1], shp[2]), dtype=np.float32)

    label_0[np.equal(label[0,:,:],0)] = 1
    label_1[np.equal(label[0,:,:],1)] = 1
    label_2[np.equal(label[0,:,:],2)] = 1
    label_12 = allone - label_0
    label_01 = allone - label_2
    label_02 = allone - label_1

    result_0[np.equal(result[0,:,:],0)] = 1
    result_1[np.equal(result[0,:,:],1)] = 1
    result_2[np.equal(result[0,:,:],2)] = 1
    result_12 = allone - result_0
    result_01 = allone - result_2
    result_02 = allone - result_1

    # tumor(2) will be defined as background for kidney segmentation
    acc1_1_tt = np.sum(label_1 * result_1)  
    acc1_1_tf =  np.sum(label_1 * result_02)
    acc1_1_ft =  np.sum(label_02 * result_1)
    # tumor(2) will be defined as kidney for kidney segmentation
    acc2_1_tt =  np.sum(label_12 * result_12)
    acc2_1_tf =  np.sum(label_12 * result_0)
    acc2_1_ft =  np.sum(label_0 * result_12)
    #only tumor will be defined as tumor for tumor segmentation
    acc_2_tt =  np.sum(label_2 * result_2)
    acc_2_tf =  np.sum(label_2 * result_01)
    acc_2_ft =  np.sum(label_01 * result_2)
    sum1_1 = acc1_1_tt * 2 + acc1_1_tf +acc1_1_ft 
    sum2_1 = acc2_1_tt * 2 + acc2_1_tf +acc2_1_ft 
    sum_2 = acc_2_tt * 2 + acc_2_tf + acc_2_ft
    
    acc1_1 = 0
    # don't divide 0 
    if sum1_1 < 0.01:
        acc1_1 = 0.5
    else:
        acc1_1 = acc1_1_tt  / sum1_1
    
    acc2_1 =0
    if sum2_1 < 0.01:
        acc2_1 = 0.5
    else:
        acc2_1 = acc2_1_tt  / sum2_1

    acc_2 =0  
    if sum_2 < 0.01:
        acc_2 = 0.5
    else:
        acc_2 = acc_2_tt / sum_2

    acc1 = acc1_1 + acc_2
    acc2 = acc2_1 + acc_2
    return acc1, acc2, acc_2*2
    
