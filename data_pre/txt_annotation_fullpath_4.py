import os
from os import getcwd
import random
from utils.utils import get_classes

'''
6.16
统计每类的样本数量 并制作随机分布的测试集
'''
# -------------------------------------------------------------------#
#   classes_path    指向model_data下的txt，与自己训练的数据集相关 
#                   训练前一定要修改classes_path，使其对应自己的数据集
#                   txt文件中是自己所要去区分的种类
#                   与训练和预测所用的classes_path一致即可
# -------------------------------------------------------------------#
classes_path = 'model_data/cls_classes.txt'
# -------------------------------------------------------#
#   datasets_path   指向数据集所在的路径
# -------------------------------------------------------#
datasets_path = 'datasets'

# sets            = ["train", "test"]
sets = ["train"]
classes, _ = get_classes(classes_path)
print(classes)
less_class = ['COVID-19', 'Emphysema', 'Fibrosis', 'Hernia', 'Mass', 'Nodule', 'PleuralOther', 'Pleural_Thickening',
              'tuberculosis']

# train_split = 0.9
valid_split = 0.2

if __name__ == "__main__":
    wd = getcwd()
    print('root path -> ', wd)

    for se in sets:
        list_file_train_small = open('cls_' + se + '_small-621.txt', 'w')
        list_file_train_all = open('cls_' + se + '_all-621.txt', 'w')
        list_file_test = open('cls_test_all-621.txt', 'w')

        datasets_path_t = os.path.join(datasets_path, se)
        types_name = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in classes:
                print('txt类出现缺失！跳过文件夹 ->', type_name)
                continue
            cls_id = classes.index(type_name)
            print('\nclass and class_id ->', type_name, cls_id)
            photos_path = os.path.join(datasets_path_t, type_name)
            photos_name = os.listdir(photos_path)
            # 随机打乱
            random.shuffle(photos_name)
            img_len = len(photos_name)
            print('all-image-len -> ', img_len)
            test_len = img_len * valid_split
            print('test_len -> ', test_len)

            if type_name not in less_class:
                max_iter = 8000
                if test_len > 5000: test_len = 5000
            else:
                max_iter = 10000
            print('test_len -> ', test_len)
            photos_iter = 0
            test_iter = 0
            globel_step = 0
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                if test_iter < test_len and globel_step % 2 == 0:
                    list_file_test.write(str(cls_id) + ";" + '%s/%s' % (wd, os.path.join(photos_path, photo_name)))
                    list_file_test.write('\n')
                    test_iter += 1
                    globel_step += 1
                else:
                    if photos_iter <= max_iter:
                        list_file_train_small.write(str(cls_id) + ";" + '%s/%s' % (wd, os.path.join(photos_path, photo_name)))
                        list_file_train_small.write('\n')
                    list_file_train_all.write(str(cls_id) + ";" + '%s/%s' % (wd, os.path.join(photos_path, photo_name)))
                    list_file_train_all.write('\n')
                    # 控制数据量
                    photos_iter += 1
                    globel_step += 1

            print('test_iter ->', test_iter)
            print('photos_iter ->', photos_iter)
            print('globel_step ->', globel_step)

        list_file_train_small.close()
        list_file_test.close()
        list_file_train_all.close()
