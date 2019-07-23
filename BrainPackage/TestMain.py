import os
import datetime 
import json
from pathlib import Path

from BrainPackage.CNN.TestCase import ParcelFCNFocusLossTest
from BrainPackage.CNN.TestCase import CalculatedforegroundRegion2


def testFCNTestKidneyFocusLossalloneForTraining( train_image_path, train_label_path, evaluation_image_path, evaluation_label_path, output_path, color_weight=[2,1,1,0], focus_loss_weight = [4,4,4]):
    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 6;
    test_input.train_image_root = train_image_path
    test_input.train_mask_root = train_label_path
    test_input.test_image_root = evaluation_image_path
    test_input.test_mask_root = evaluation_label_path
    test_input.data_size = 192
    test_input.echo_number = 11
    test_input.imagetype = r'rough'
    test_input.stamp = r'rough_output'
    test_input.saved_train_data_name =  'traindata.txt'
    test_input.saved_test_data_name = 'testdata.txt'
    test_input.checkpoint_name = ''
    test_input.thread_number =2
    print(test_input.thread_number)
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point = False, is_random_select = 'fix', net_type = 'FCN',
                                                                segment_number = 3, weight = color_weight, focus_weight = focus_loss_weight)
    test_case.PerformTestCase()

def testFCNTestKidneyFocusLossalloneForOutputTrainData(train_image_path, train_label_path, output_path, check_point_name, color_weight=[2,1,1,0], focus_loss_weight = [4,4,4]):
    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 4;
    test_input.train_image_root = train_image_path
    test_input.train_mask_root = train_label_path
    test_input.test_image_root = train_image_path
    test_input.test_mask_root = train_label_path
    test_input.data_size = 192
    test_input.echo_number = 1
    test_input.imagetype = r'rough'
    test_input.stamp = r'rough_train_data'
    test_input.saved_train_data_name =  'traindata2.txt'
    test_input.saved_test_data_name = 'testdata2.txt'
    test_input.thread_number =2
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    test_input.checkpoint_name = check_point_name
    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point = True, is_random_select = 'fix', net_type = 'FCN', 
     segment_number = 3, weight = color_weight, focus_weight = focus_loss_weight, is_only_test = True)
    test_case.PerformTestCase()

def testFCNTestKidneyFocusLossalloneForTest(test_image_path, output_path, check_point_name, color_weight = [2,1,1,0], focus_loss_weight = [4,4,4]):
    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 4;
    test_input.train_image_root = test_image_path
    test_input.train_mask_root = test_image_path
    test_input.test_image_root = test_image_path
    test_input.test_mask_root = test_image_path
    test_input.data_size = 192
    test_input.echo_number = 1
    test_input.imagetype = r'rough'
    test_input.stamp = r'rough_test'
    test_input.saved_train_data_name =  'traindata2.txt'
    test_input.saved_test_data_name = 'testdata2.txt'
    test_input.thread_number =2
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    test_input.checkpoint_name = check_point_name
    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point = True, is_random_select = 'fix', net_type = 'FCN', 
     segment_number = 3, weight = color_weight, focus_weight = focus_loss_weight, is_only_test = True)
    test_case.PerformTestCase()

def testFCNTestKidneyFocusLossalloneForBothKindneyDetailedImageForTrain(train_image_path_left, train_label_path_left, evaluation_image_path_left, evaluation_label_path_left,
                                                                        train_image_path_right, train_label_path_right, evaluation_image_path_right, evaluation_label_path_right, output_path,
                                                                        color_weight=[2,1,1,0], focus_loss_weight = [4,4,4]):
    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 5;
    test_input.train_image_root =  train_image_path_left
    test_input.train_mask_root = train_label_path_left
    test_input.test_image_root =  evaluation_image_path_left
    test_input.test_mask_root = evaluation_label_path_left
    test_input.data_size = 192
    test_input.echo_number = 10
    test_input.stamp = 'left'
    test_input.imagetype = 'left'

    test_input.saved_train_data_name =  'left_traindata.txt'
    test_input.saved_test_data_name = 'left_testdata.txt'
    test_input.checkpoint_name = ''
    test_input.thread_number = 2
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    print('left_kidney')

    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point=False,
                                                              is_random_select='fix', net_type='FCN', segment_number=3,
                                                              weight=color_weight, focus_weight=focus_loss_weight)
    test_case.PerformTestCase()

    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 5;
    test_input.train_image_root = train_image_path_right
    test_input.train_mask_root = train_label_path_right
    test_input.test_image_root = evaluation_image_path_right
    test_input.test_mask_root = evaluation_label_path_right
    test_input.data_size = 192
    test_input.echo_number = 10
    test_input.stamp = 'right'
    test_input.imagetype = 'right'
    test_input.saved_train_data_name =  'right_train_data.txt'
    test_input.saved_test_data_name = 'right_test_ata.txt'
    # test_input.checkpoint_name = 'stamp_small_batch_r_FCN_model_3.pt'
    test_input.checkpoint_name = ''
    test_input.thread_number = 2
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    print('right _kidney')
    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point=False,
                                                              is_random_select='fix', net_type='FCN', segment_number=3,
                                                              weight=color_weight, focus_weight=focus_loss_weight)
    test_case.PerformTestCase()

def testFCNTestKidneyFocusLossalloneForBothKindneyDetailedImageForTest(test_image_path_left, checkpoint_name_left,
                                                                        test_image_path_right, checkpoint_name_right, output_path,
                                                                        color_weight=[2,1,1,0], focus_loss_weight = [4,4,4]):
    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 5;
    test_input.train_image_root =  test_image_path_left
    test_input.train_mask_root = test_image_path_left
    test_input.test_image_root =  test_image_path_left
    test_input.test_mask_root = test_image_path_left
    test_input.data_size = 192
    test_input.echo_number = 10
    test_input.stamp = 'left_test'
    test_input.imagetype = 'left'

    test_input.saved_train_data_name =  'left_traindata.txt'
    test_input.saved_test_data_name = 'left_testdata.txt'
    test_input.checkpoint_name = checkpoint_name_left
    test_input.thread_number = 2
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    print('left_kidney')

    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point=True,
                                                              is_random_select='fix', net_type='FCN', segment_number=3,
                                                              weight=color_weight, focus_weight=focus_loss_weight,is_only_test=True)
    test_case.PerformTestCase()

    test_input = ParcelFCNFocusLossTest.SegmentTestInfo
    test_input.batch_size = 5;
    test_input.train_image_root = test_image_path_right
    test_input.train_mask_root = test_image_path_right
    test_input.test_image_root = test_image_path_right
    test_input.test_mask_root = test_image_path_right
    test_input.data_size = 192
    test_input.echo_number = 10
    test_input.stamp = 'right_test'
    test_input.imagetype = 'right'
    test_input.saved_train_data_name =  'right_train_data.txt'
    test_input.saved_test_data_name = 'right_test_ata.txt'
    # test_input.checkpoint_name = 'stamp_small_batch_r_FCN_model_3.pt'
    test_input.checkpoint_name = checkpoint_name_right
    test_input.thread_number = 2
    test_input.output_path = output_path
    if not os.path.exists(output_path):
         os.mkdir(output_path)
    print('right _kidney')
    test_case = ParcelFCNFocusLossTest.ParcelFCNFocusLossTest(test_input, is_load_check_point=True,
                                                              is_random_select='fix', net_type='FCN', segment_number=3,
                                                              weight=color_weight, focus_weight=focus_loss_weight,is_only_test=True)
    test_case.PerformTestCase()

def testCalculateForeGround2(mask_root, result_root, output_root, original_image_root, original_mask_root, is_not_test = True):
    test_input = CalculatedforegroundRegion2.SegmentTestInfo
    test_input.test_result_root = result_root
    test_input.test_mask_root = mask_root
    test_input.saved_image_root = output_root
    test_input.original_image_root = original_image_root
    test_input.original_label_root = original_mask_root
    test_input.stamp = 'w'
    test_input.is_not_test = is_not_test;
    test_case = CalculatedforegroundRegion2.CalculatedforegroundRegion2(test_input)
    test_case.PerformTestCase()

