
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from utils.utils import letterbox_image
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)
import matplotlib.pyplot as plt  # plt 用于显示图片
from PIL import Image
from tqdm import tqdm

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _preprocess_input(x, ):
    x /= 127.5
    x -= 1.
    return x


def load_img_mai(img_path, input_shape):
    # old_image = copy.deepcopy(image)
    # ---------------------------------------------------#
    #   对图片进行不失真的resize
    # ---------------------------------------------------#
    # try:
    #     image = Image.open(img_path)
    # except:
    #     print('Error: cant not load image file :', img_path)

    image = Image.open(img_path)
    image = cvtColor(image)
    crop_img = letterbox_image(image, [input_shape[0], input_shape[1]], False)
    image_data = np.expand_dims(preprocess_input(np.array(crop_img, np.float32)), 0)
    # photo = np.array(crop_img, dtype=np.float32)
    # photo = np.reshape(_preprocess_input(photo), [1, input_shape[0], input_shape[1], input_shape[2]])
    return image_data


def gradient_compute(model, layername, img, input_index_flag=True, label_index: int = None):
    preds = model.predict(img)
    if not input_index_flag:
        idx = np.argmax(preds[0])  # 返回预测图片最大可能性的index索引
    else:
        idx = label_index
    # print(preds[0])
    print(idx, '-> class_index索引')

    output = model.output[:, idx]  # 获取到我们对应索引的输出张量
    last_layer = model.get_layer(layername)

    grads = K.gradients(output, last_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 对每张梯度特征图进行平均，
    # 返回的是一个大小是通道维数的张量
    iterate = K.function([model.input], [pooled_grads, last_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img])

    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    return conv_layer_output_value


def plot_heatmap(conv_layer_output_value, img_in_path, img_out_path, vis=False):
    """
    绘制热力图
    :param conv_layer_output_value: 卷积层输出值
    :param img_in_path: 输入图像的路径
    :param img_out_path: 输出热力图的路径
    :return:
    """
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_in_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimopsed_img = heatmap * 0.4 + img
    # print(superimopsed_img)

    cv2.imwrite(img_out_path, superimopsed_img)
    if vis == True:
        img = cv2.imread(img_out_path)
        cv2.imshow("heatmap", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def heatmap(vis=True):
    '''
    输出指定特征层热力图
    '''

    img_path = r'test_img/ori_tuberculosis001.jpg'

    layername = r'activation_48'

    from classification import Classification
    classfication = Classification()
    model = classfication.model

    img = load_img_mai(img_path, [512, 512, 3])

    conv_value = gradient_compute(model, layername, img)
    # print(conv_value)
    plot_heatmap(conv_value, img_path, 'heatmap_output/' + layername + '.png', vis)


def heatmap_putall():
    '''
    输出所有特征层的热力图
    '''
    from classification import Classification
    classfication = Classification()
    model = classfication.model

    img_path = r''

    img = load_img_mai(img_path, [512, 512, 3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        # if 133 < i <= 174:  # resnet输出后段的特征层
        if 0 < i <= 174:  # resnet输出后段的特征层
            # if 50 < i <= 87:  # moblienet输出后段的特征层
            conv_value = gradient_compute(model, layer.name, img, False, 3)
            plot_heatmap(conv_value, img_path, 'heatmap_output/test/' + str(i) + '-' + layer.name + '.png')
        else:
            pass

    '''
    167-res5c_branch2b
    149-res5a_branchnormal
    173-activation_49
    '''


def heatmap_range(test_annotation_path):
    '''
    输出测试集的grad-cam，分类别格式保存热力图
    Args:
        test_annotation_path:
    Returns:
    '''
    import os
    from classification import Classification
    classfication = Classification()
    model = classfication.model
    layername = r'activation_48'

    with open(test_annotation_path, "r") as f:
        lines = f.readlines()
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]

        class_name, file_name = os.path.split(annotation_path)

        class_name = class_name.split("\\")[-1]

        label_true = int(line.split(';')[0])

        img_path = annotation_path
        img = load_img_mai(img_path, [512, 512, 3])

        conv_value = gradient_compute(model, layername, img, True, label_true)
        # print(conv_value)
        save_dest = 'heatmap_output/' + class_name + '/'
        if not os.path.exists(save_dest):
            os.makedirs(save_dest)
        img_save_path = save_dest + file_name
        print(img_save_path)
        plot_heatmap(conv_value, img_path, img_save_path, vis=False)


def heatmap_range_txt(test_annotation_path, classes_path, output_dest, max_ouput):
    '''
    根据txt path 输出grad-cam，分类别格式保存热力图
    需要读取txt中的类别index 获得对应类名str
    加入了控制数量dict
    Args:
        test_annotation_path:
        max_ouput: 最大生成数量
    Returns:
    '''
    import os
    from classification import Classification
    classfication = Classification()
    model = classfication.model
    # 获取类名
    classes_path = classes_path
    class_names, num_classes = get_classes(classes_path)

    output_count: dict = {}

    layername = r'activation_48'

    with open(test_annotation_path, "r") as f:
        lines = f.readlines()

    # for index, line in enumerate(lines):
    for index, line in tqdm(enumerate(lines), total=len(lines)):

        annotation_path = line.split(';')[1].split()[0]

        _, file_name = os.path.split(annotation_path)

        label_true = int(line.split(';')[0])
        # print(label_true)
        class_name = class_names[label_true]
        # print(class_name)
        img_path = annotation_path
        if output_count.get(label_true) is None:
            output_count[label_true] = 0

        output_count[label_true] += 1

        if output_count[label_true] > max_ouput:
            pass
        else:
            try:
                img = load_img_mai(img_path, [512, 512, 3])
            except:
                print('Error: cant not load image file :', img_path)
                continue

            conv_value = gradient_compute(model, layername, img, True, label_true)

            # save_dest = 'heatmap_output/' + class_name + '/'
            save_dest = output_dest + class_name + '/'
            if not os.path.exists(save_dest):
                os.makedirs(save_dest)
            img_save_path = save_dest + file_name
            # print(img_save_path)

            plot_heatmap(conv_value, img_path, img_save_path, vis=False)

    print(output_count)


if __name__ == '__main__':
    # 数据集路径path文件
    test_annotation_path = 'data_path_txt/data_path_vindr-pcxr_train.txt'
    # 获取类名
    classes_path = 'model_data/cls_classes.txt'
    output_dest = 'heatmap_output/ressult1/'
    heatmap_range_txt(test_annotation_path, classes_path, output_dest, 100)
