import os
import random
from BrainPackage.Exception import ProgramException

def GetRandomFileListByPercentList(
    folder_path, 
    percentlist,
    data_folder_structure = [['NC',1], ['AD',0]]
):
    data = []
    labels = []
    for index in range(len(percentlist)):
        tempdata = []
        templabels = []
        sort_dict = []
        data.append(tempdata)
        labels.append(templabels)
    for sub_folder in data_folder_structure:
            folder_name = sub_folder[0]
            folder_tagert_value = sub_folder[1]
            sub_folder_full_path  = os.path.join(folder_path, folder_name)
            if not os.path.exists(sub_folder_full_path):
                raise ProgramException('ParcelDataSet','The following sub folder is not exist:\n' + sub_folder_full_path)
            try:
               file_list = sorted(os.listdir(sub_folder_full_path))
            except:
                raise ProgramException('ParcelDataSet','The following folder has unexpected sub folder:\n' + sub_folder_full_path)
            input_number = len(file_list)
            if input_number <= 0:
                raise ProgramException('ParcelDataSet', 'The following sub folder has no file:\n' + sub_folder_full_path)
            for file in file_list:
                file_full_path = os.path.join(sub_folder_full_path, file)
                file_name = file.split(".")[0]
                data_name = file_name.split("_")[1]               
                if not os.path.exists(file_full_path):
                     raise ProgramException('ParcelDataSet', 'The following file is not exist:\n' + file_full_path)
                randomNumber = random.uniform(0, 10000)
                sort_dict.append((randomNumber, data_name, folder_tagert_value))
                #data[currentLevel].append(data_name)
                #labels[currentLevel].append(folder_tagert_value)
    result = sorted(sort_dict, key=lambda x: x[0])
    for index in range(len(result)):
        currentLevel = 0
        for randomLevel in range(len(percentlist) - 1): 
            if index > sum(percentlist[0:randomLevel + 1] * len(result)):
                currentLevel = randomLevel + 1
        data[currentLevel].append(result[index][1])
        labels[currentLevel].append(result[index][2])
    return data, labels
def GetRandomFileListByPercentListForSegment(
    folder_path, 
    percentlist
):
    data = []
    sort_dict = []
    for index in range(len(percentlist)):
        tempdata = []
        data.append(tempdata)

    if not os.path.exists(folder_path):
        raise ProgramException('ParcelDataSet','The following sub folder is not exist:\n' + folder_path)
    try:
        file_list = sorted(os.listdir(folder_path))
    except:
        raise ProgramException('ParcelDataSet','The following folder has unexpected sub folder:\n' + folder_path)
    input_number = len(file_list)
    if input_number <= 0:
        raise ProgramException('ParcelDataSet', 'The following sub folder has no file:\n' + folder_path)
    for file in file_list:
        randomNumber = random.uniform(0, 10000)
        sort_dict.append((randomNumber, file))
    result = sorted(sort_dict, key=lambda x: x[0])
    for index in range(len(result)):
        currentLevel = 0
        for randomLevel in range(len(percentlist) - 1): 
            if index > sum(percentlist[0:randomLevel + 1] * len(result)):
                currentLevel = randomLevel + 1
        data[currentLevel].append(result[index][1])
    return data
