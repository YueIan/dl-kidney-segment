from abc import ABCMeta, abstractmethod
import os
import time
import datetime
import torch
import numpy as np
from BrainPackage.Exception import ProgramException
from BrainPackage.CNN.Utility import ParcelHelper

class ParcelTestInfo:
    '''Input structure'''
    train_dataset_root=''
    test_dataset_root=''
    parcel_info_path=r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\log\parcel_range_result_100.txt'
    log_path=''
    result_path=''
    intermediate_result =r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Intermediate_Result'
    batch_size = 10
    epoch = 80
    parcel_name = 'Hippocampus'
    is_display_encode_result = False
    encode_model_path = None
    check_value = 0
class FusionParcelTestInfo:
    train_dataset_root=''
    test_dataset_root=''
    parcel_info_path=r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\log\parcel_range_result_100.txt'
    log_path=''
    intermediate_result =r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Intermediate_Result'
    batch_size = 10
    epoch = 80
    check_value = 0
class ParcelTestBase:
    '''The base class of parcel test case'''

    __metaclass__=ABCMeta

    def __init__(self, test_info):
        self.CheckGPU()
        self.train_dataset_root = test_info.train_dataset_root
        self.test_dataset_root = test_info.test_dataset_root
        self.parcel_info_path = test_info.parcel_info_path
        self.log_path = test_info.log_path
        self.check_value = test_info.check_value
        self.result_path = test_info.result_path
        self.loss_point_test = []
        self.accuracy_point_test = []
        self.loss_point = []
        self.accuracy_point = []
        self.log_full_path = self.__prepareTrainLogfile(test_info)
              
    def PerformTestCase(self):
        self.PrepareTestCase()
        self.RunTestCase()
    
    @abstractmethod
    def PrepareTestCase(self):
        pass

    @abstractmethod
    def RunTestCase(self):
        pass

    def CheckGPU(self):
        cuda_gpu = torch.cuda.is_available()
        if (cuda_gpu == False):
            raise ProgramException('ParcelTestBase',
                'Cannot use GPU!'
            )

    def __prepareTrainLogfile(self,test_info):
        log_path = test_info.log_path
        log_file_name = 'Train_log_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
        log_full_Path = os.path.join(log_path,log_file_name)
        return log_full_Path


    def getParcelInfo(self):
        try:
            parcel_folder = self.train_dataset_root.split("\\")[-2]
            parcel_name = parcel_folder.split("_")[-1]
            self.parcel_name = parcel_name
        except:
            raise ProgramException('ParcelTestCase::getParcelInfo',
                'Cannot parse parcel name from data path'
            )
        return ParcelHelper.getParcelInfoFromLogFile(parcel_name, self.parcel_info_path)
    
    def saveResult(self, score_result, score_target):
        time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        output_path = os.path.join(self.result_path, str(time_str))
        os.makedirs(output_path)
        np.savetxt(os.path.join(output_path, 'train_FullConnect_loss'+ self.parcel_name + '_'+str(time_str)+'.txt'), self.loss_point)
        np.savetxt(os.path.join(output_path, 'test_FullConnect_loss'+ self.parcel_name + '_'+str(time_str)+'.txt'), self.loss_point_test)
        np.savetxt(os.path.join(output_path, 'train_FullConnect_acc'+ self.parcel_name + '_'+str(time_str)+'.txt'), self.accuracy_point)
        np.savetxt(os.path.join(output_path, 'test_FullConnect_acc'+ self.parcel_name + '_'+str(time_str)+'.txt'), self.accuracy_point_test)
        np.savetxt(os.path.join(self.log_path, 'score_result'+ self.parcel_name + '_'+str(time_str)+'.txt'), score_result)
        np.savetxt(os.path.join(self.log_path, 'score_target'+ self.parcel_name + '_'+str(time_str)+'.txt'), score_target)   


