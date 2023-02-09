'''
# 數據預處理 @homxxx
# 读取csv标签进行chexpert数据集的批量处理
'''
import sys
# 将终端 Terminal 或者控制台的输出结果输出至 log 文件 以文件形式保存
class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("data_chexpert.log")

import xlrd
import os
import shutil

global iter
global miss_iter
miss_iter = 0
iter = 0
import os
# import panda

# 读取数据
import pandas as pd

data = pd.read_csv("CheXpert-v1.0-small/train.csv")

# print('len(data) -> ', len(data))
'len(data) ->  223414'
# print('data.head() -> ', data.head())

Class_dic = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
print('len(Class) -> ', len(Class_dic))
# len class =13


def copy_allfiles(full_file_name, dest):
    '''
    :param full_file_name: 原路径
    :param dest: 目标文件夹
    :return:
    '''
    global iter
    global miss_iter
    # copy
    # # shutil.copy(full_file_name, dest + '/' + 'CheXpert' +str(iter) + '.jpg')
    # print(full_file_name, '->', dest)
    # iter = iter + 1

    if os.path.isfile(full_file_name):
        print('iter -> ', iter)
        # copy
        # shutil.copy(full_file_name, dest + '/' + 'CheXpert' +str(iter) + '.jpg')
        print(full_file_name, '->', dest)
        iter = iter + 1
    else:
        print('does not exit!')
        miss_iter = miss_iter + 1



def copy_in_csv(source_path, Class):
    '''
    :param source_path:
    :param Class:
    :return:
    '''
    class_info = Class
    target_path = 'class_file/' + class_info
    copy_allfiles(source_path, target_path)


# def traverse_csv_mimic():
#     '''
#     遍历csv
#     :return: null
#     '''
#     for i in range(len(data)):
#         if data['Atelectasis'][i] == 1.0:
#             print('Atelectasis : ', data['subject_id'][i], data['study_id'][i], data['Atelectasis'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Atelectasis')
#         if data['Cardiomegaly'][i] == 1.0:
#             print('Cardiomegaly : ', data['subject_id'][i], data['study_id'][i], data['Cardiomegaly'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Cardiomegaly')
#         if data['Consolidation'][i] == 1.0:
#             print('Consolidation : ', data['subject_id'][i], data['study_id'][i], data['Consolidation'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Consolidation')
#         if data['Edema'][i] == 1.0:
#             print('Edema : ', data['subject_id'][i], data['study_id'][i], data['Edema'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Edema')
#         if data['Enlarged Cardiomediastinum'][i] == 1.0:
#             print('Enlarged Cardiomediastinum : ', data['subject_id'][i], data['study_id'][i],
#                   data['Enlarged Cardiomediastinum'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Enlarged Cardiomediastinum')
#
#         if data['Fracture'][i] == 1.0:
#             print('Fracture : ', data['subject_id'][i], data['study_id'][i], data['Fracture'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Fracture')
#         if data['Lung Lesion'][i] == 1.0:
#             print('Lung Lesion : ', data['subject_id'][i], data['study_id'][i], data['Lung Lesion'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Lung Lesion')
#         if data['Lung Opacity'][i] == 1.0:
#             print('Lung Opacity : ', data['subject_id'][i], data['study_id'][i], data['Lung Opacity'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Lung Opacity')
#         if data['Pleural Effusion'][i] == 1.0:
#             print('Pleural Effusion : ', data['subject_id'][i], data['study_id'][i], data['Pleural Effusion'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Pleural Effusion')
#         if data['Pneumonia'][i] == 1.0:
#             print('Pneumonia : ', data['subject_id'][i], data['study_id'][i], data['Pneumonia'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Pneumonia')
#
#         if data['Pneumothorax'][i] == 1.0:
#             print('Pneumothorax : ', data['subject_id'][i], data['study_id'][i], data['Pneumothorax'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Pneumothorax')
#         if data['Pleural Other'][i] == 1.0:
#             print('Pleural Other : ', data['subject_id'][i], data['study_id'][i], data['Pleural Other'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Pleural Other')
#         if data['Support Devices'][i] == 1.0:
#             print('Support Devices : ', data['subject_id'][i], data['study_id'][i], data['Support Devices'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'Support Devices')
#         if data['No Finding'][i] == 1.0:
#             print('No Finding : ', data['subject_id'][i], data['study_id'][i], data['No Finding'][i])
#             copy_in_csv(data['subject_id'][i], data['study_id'][i], 'No Finding')
#
#     print(i)
#     print(iter, 'files done!')


def traverse_csv_chexpert():
    '''
    遍历csv
    :return: null
    '''
    for i in range(len(data)):
        for name in Class_dic:
            if data[name][i] == 1.0:
                print(i, name, data['Path'][i])
                copy_in_csv(data['Path'][i], name)
    print(i)

    print(iter, 'files done!')
    print(miss_iter, 'files miss!')


if __name__ == '__main__':
    print('begin...')
    traverse_csv_chexpert()

    import time
    localtime = time.asctime(time.localtime(time.time()))
    print("本地时间为 :", localtime)
