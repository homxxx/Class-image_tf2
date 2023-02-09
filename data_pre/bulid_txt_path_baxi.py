'''
@homxx
brax数据集处理
提取csv建立txt路径
'''
import os
from os import getcwd
import sys
import pandas as pd

data = pd.read_csv("master_spreadsheet_update-mylabel.csv")

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

classes_path = 'cls_classes.txt'

classes, _ = get_classes(classes_path)

csv_class = []

def traverse_csv_chexpert():
    '''
    遍历csv
    :return: null
    '''
    for i in range(len(data)):
        for name in classes:
            try:
                # if data['SupportDevices'][i] == 1 and data['NoFindingorNormal'][i] == 1:
                #     absolu_path = "img\\" + data['PngPath'][i].split("/")[-1]
                #     print(i, absolu_path)
                #     label = 'SupportDevices'
                #     cls_id = classes.index(label)
                #     path_file.write(str(cls_id) + ";" + '%s\%s' % (root_path, absolu_path))
                if data[name][i] == 1:
                    # print(i, name, data['PngPath'][i])
                    absolu_path = "img\\" + data['PngPath'][i].split("/")[-1]
                    print(i, absolu_path)
                    label = name
                    cls_id = classes.index(label)
                    path_file.write(str(cls_id) + ";" + '%s\%s' % (root_path, absolu_path))
                    path_file.write('\n')
            except:
                pass
                # print('missing labe',name)
    print(i)


if __name__ == '__main__':
    root_path = getcwd()
    print('root path -> ', root_path)

    path_file = open('data_path_brax.txt', 'w')

    traverse_csv_chexpert()

    import time
    localtime = time.asctime(time.localtime(time.time()))
    print("本地时间为 :", localtime)

    path_file.close()
