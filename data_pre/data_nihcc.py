'''
nihcc数据集
'''
miss_iter = 0
iter = 0
import os
import shutil
import pandas as pd
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

sys.stdout = Logger("data_nihcc.log")


data = pd.read_csv("Data_Entry_2017.csv")
print('len(data) -> ', len(data))

def copy_allfiles(full_file_name, dest, class_info):
    '''
    :param full_file_name: 原路径
    :param dest: 目标文件夹
    :return:
    '''
    global iter
    global miss_iter


    if os.path.isfile(full_file_name):
        target_name = dest + '/nih_' + class_info.replace(" ", "") +str(iter) + '.jpg'
        # copy
        # shutil.copy(full_file_name, dest + '/nih_' + class_info +str(iter) + '.jpg')
        shutil.copy(full_file_name, target_name)
        print(full_file_name, '->', target_name)
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
    target_path = 'calss_file/' + class_info
    source_path = 'preclass_file/' + source_path
    copy_allfiles(source_path, target_path, class_info)

# (1, Atelectasis; 2, Cardiomegaly; 3, Effusion; 4, Infiltration; 5, Mass; 6, Nodule; 7, Pneumonia; 8,
# Pneumothorax; 9, Consolidation; 10, Edema; 11, Emphysema; 12, Fibrosis; 13,
# Pleural_Thickening; 14 Hernia)

def traverse_csv_nih():
    '''
    遍历csv
    :return: null
    '''
    global iter
    global miss_iter

    for i in range(len(data)):
        # class_name = data['Finding Labels'][i]
        class_name = data['Finding Labels'][i].split('|')
        for name in class_name:
            print(i, name, data['Image Index'][i])
            copy_in_csv(data['Image Index'][i], name)

    print(i)

    print(iter, 'files done!')
    print(miss_iter, 'files miss!')


if __name__ == '__main__':
    print('begin...')
    traverse_csv_nih()

    import time
    localtime = time.asctime(time.localtime(time.time()))
    print("本地时间为 :", localtime)