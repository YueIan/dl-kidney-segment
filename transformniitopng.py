from starter_code.utils import load_case
from starter_code.visualize import visualize
from starter_code.visualizetestdata import visualizetest
import os
from pathlib import Path
import numpy as np
import json
import shutil

def GenerateTrainpng(input_path, output_path):
    for i in range(160):
        name = "case_{:05d}".format(i)
        visualize(name, input_path, output_path) 

def GenerateEvaluationpng(input_path, output_path):
    for i in range(50):
        name = "case_{:05d}".format(i+160)
        visualize(name, input_path, output_path)    
def GenerateTestpng(input_path, output_path):
    for i in range(90):
        name = "case_{:05d}".format(i+210)
        visualizetest(name, input_path, output_path)