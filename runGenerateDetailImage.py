from BrainPackage.TestMain import testCalculateForeGround2

def GenerateDetailTrainingImage(mask_root, result_root, output_root, original_image_root, original_mask_root):
    testCalculateForeGround2(mask_root, result_root, output_root, original_image_root, original_mask_root)
def GenerateDetailEvaluationImage(mask_root, result_root, output_root, original_image_root, original_mask_root):
    testCalculateForeGround2(mask_root, result_root, output_root, original_image_root, original_mask_root)
def GenerateDetailTestImage(result_root, output_root, original_image_root):
    testCalculateForeGround2(result_root, result_root, output_root, original_image_root, original_image_root,False)