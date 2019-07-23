from transformniitopng import GenerateTrainpng
from transformniitopng import GenerateEvaluationpng
from transformniitopng import GenerateTestpng
from outputnii import output_result_data_nii
from outputnii import output_evaluation_data_nii_and_get_score
from runGenerateDetailImage import GenerateDetailEvaluationImage
from runGenerateDetailImage import GenerateDetailTrainingImage
from runGenerateDetailImage import GenerateDetailTestImage
from BrainPackage.TestMain import testFCNTestKidneyFocusLossalloneForTraining
from BrainPackage.TestMain import testFCNTestKidneyFocusLossalloneForOutputTrainData
from BrainPackage.TestMain import testFCNTestKidneyFocusLossalloneForTest
from BrainPackage.TestMain import testFCNTestKidneyFocusLossalloneForBothKindneyDetailedImageForTrain
from BrainPackage.TestMain import testFCNTestKidneyFocusLossalloneForBothKindneyDetailedImageForTest
import os
if __name__ == '__main__':

    input_path = r'E:\kidney_data_backup\kits19'
    output_path =  r'D:\kidney20190718'
    train_data_output_path = r'D:\kidney20190718\train'
    evaluation_data_output_path = r'D:\kidney20190718\evaluation'
    test_data_output_path = r'D:\kidney20190718\test'

    detail_segment_image_size = 192

    # 1 generate rough segmentaton training, evaluation test data from nii file
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if os.path.exists(train_data_output_path):
        os.path.mkdir(train_data_output_path)
    if os.path.exists(evaluation_data_output_path):
        os.path.mkdir(evaluation_data_output_path)
    if os.path.exists(test_data_output_path):
        os.path.mkdir(test_data_output_path)
    GenerateTrainpng(input_path, train_data_output_path)
    GenerateEvaluationpng(input_path, evaluation_data_output_path)
    GenerateTestpng(input_path, test_data_output_path)

    # 2 rough train FCN dilation model, output evaluation image

    rough_train_data_image_path = os.path.join(train_data_output_path,'image')
    rough_train_data_label_path = os.path.join(train_data_output_path,'label')
    # # rough_segment_training_label_path
    rough_evaluation_data_image_path = os.path.join(evaluation_data_output_path,'image')
    rough_evaluation_data_label_path = os.path.join(evaluation_data_output_path,'label')
    # # rough_segment_evaluation_label_path
    rough_evaluation_output_path = os.path.join(output_path,'output_evaluation_image')
    # # rough_segment_evaluation_path
    #testFCNTestKidneyFocusLossalloneForTraining(rough_train_data_image_path, rough_train_data_label_path, rough_evaluation_data_image_path, rough_evaluation_data_label_path, routh_evaluation_output_path)
    # 3get rough test result, output test and train image

    rough_check_point_name = r'stamp_rough_FCN_model_10.plt'
    rough_train_output_path = os.path.join(output_path,'output_train_image')
    # # rough_segment_training_path
    testFCNTestKidneyFocusLossalloneForOutputTrainData(rough_train_data_image_path, rough_train_data_label_path, rough_train_output_path, rough_check_point_name)
    #
    rough_test_output_path = os.path.join(output_path,'oupuy_test_image')
    # error
    rough_test_data_image_path = os.path.join(test_data_output_path, 'image')
    testFCNTestKidneyFocusLossalloneForTest(rough_test_data_image_path, rough_test_output_path, rough_check_point_name)

    # rough_segment_result_path


    # 4get left and right kidney from rough train result(include training data, evaluation data and test data)

    original_image_root = train_data_output_path + r'\image'
    original_mask_root = train_data_output_path + r'\label'
    rough_segment_training_label_path = rough_train_output_path + r'\label\rough'
    rough_segment_training_result_path = rough_train_output_path + r'\result\rough'
    GenerateDetailTrainingImage(rough_segment_training_label_path, rough_segment_training_result_path, train_data_output_path, original_image_root, original_mask_root)
    original_image_root = evaluation_data_output_path + r'\image'
    original_mask_root = evaluation_data_output_path + r'\label'
    rough_segment_evaluation_label_path = rough_evaluation_output_path + r'\label\rough'
    rough_segment_evaluation_path = rough_evaluation_output_path + r'\result\rough'
    GenerateDetailEvaluationImage(rough_segment_evaluation_label_path, rough_segment_evaluation_path, evaluation_data_output_path, original_image_root, original_mask_root)
    #
    original_image_root = test_data_output_path + r'\image'
    rough_segment_result_path = rough_test_output_path+ r'\result\rough'
    GenerateDetailTestImage(rough_segment_result_path, test_data_output_path, original_image_root)

    # 5train left and right FCN dilation model

    left_train_data_image_path = os.path.join(train_data_output_path,'limage')
    left_train_data_label_path = os.path.join(train_data_output_path,'llabel')
    right_train_data_image_path = os.path.join(train_data_output_path, 'rimage')
    right_train_data_label_path = os.path.join(train_data_output_path, 'rlabel')

    left_evaluation_data_image_path = os.path.join(evaluation_data_output_path, 'limage')
    left_evaluation_data_label_path = os.path.join(evaluation_data_output_path, 'llabel')
    right_evaluation_data_image_path = os.path.join(evaluation_data_output_path, 'rimage')
    right_evaluation_data_label_path = os.path.join(evaluation_data_output_path, 'rlabel')
    #testFCNTestKidneyFocusLossalloneForBothKindneyDetailedImageForTrain(left_train_data_image_path,left_train_data_label_path,left_evaluation_data_image_path,left_evaluation_data_label_path, right_train_data_image_path, right_train_data_label_path,right_evaluation_data_image_path,right_evaluation_data_label_path,routh_evaluation_output_path)

    # 6get test result
    left_check_point_name = r'stamp_left_FCN_model_9.plt'
    right_check_point_name = r'stamp_right_FCN_model_9.plt'
    left_test_data_image_path = os.path.join(test_data_output_path, 'limage')
    right_test_data_image_path = os.path.join(test_data_output_path, 'rimage')
    testFCNTestKidneyFocusLossalloneForBothKindneyDetailedImageForTest(left_test_data_image_path, left_check_point_name,
                                                                        right_test_data_image_path, right_check_point_name,
                                                                        rough_test_output_path)
    # 7generate nii file based on rough test result and test result

    detail_segment_information_result_path = os.path.join(test_data_output_path, 'information3.csv')
    detail_segment_result_path = rough_test_output_path + r'\result'
    resultnii_output_path = rough_test_output_path+ r'\result\output'
    output_result_data_nii(rough_segment_result_path, detail_segment_result_path, detail_segment_information_result_path, detail_segment_image_size, resultnii_output_path)

    detail_segment_information_evaluation_filename = os.path.join(evaluation_data_output_path, 'information3.csv')
    detail_segment_evaluation_path = rough_evaluation_output_path + r'\result'
    evaluation_nii_output_path = rough_evaluation_output_path + r'\result\output'
    detail_segment_information_evaluation_path = os.path.join(detail_segment_evaluation_path, detail_segment_information_evaluation_filename)
    output_evaluation_data_nii_and_get_score(input_path, rough_segment_evaluation_path, detail_segment_evaluation_path,
     detail_segment_information_evaluation_path, detail_segment_image_size, evaluation_nii_output_path)