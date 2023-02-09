'''
auc mask eval
区域masked auc贡献率
包含未经过mask 与 侧位单独的auc；
一共20份生成数据；
'''
import csv
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
from PIL import Image
from classification import Classification
from classification_front import Classificationfront
from PIL import Image
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_curve
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.ticker import FuncFormatter
from scipy import interp
import tensorflow as tf
from tqdm import tqdm
from add_mask import mask_list, add_mask
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def write_csv(metrics_out_path, test_annotation_path, root_path, ii, mask):
    '''
    读取测试集列表进行测试并输出数据结果保存csv
    :return:
    '''
    # 实例化模型
    classfication = Classification()
    classfication_front = Classificationfront()

    print('test_annotation_path - > ', test_annotation_path)

    # 1. 创建文件对象
    f = open(metrics_out_path + '/roc_eval.csv', 'w', newline='')
    # f_lateral追加模式 以a+的方式打开
    # f_lateral = open(root_path + '/Lateral/roc_eval.csv', 'a+', newline='')
    # 只需写入一次
    f_lateral = open(root_path + '/Lateral/roc_eval.csv', 'w', newline='')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    csv_lateral = csv.writer(f_lateral)

    # 3. 构建列表头
    tabel_head_list = ["image_path", "y_real", "y_predict"]
    tabel_head_list += class_names
    csv_writer.writerow(
        tabel_head_list
    )
    csv_lateral.writerow(
        tabel_head_list
    )

    # 4. 调用模型写入csv文件内容
    with open(test_annotation_path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    for index, line in tqdm(enumerate(lines), total=total):
        annotation_path = line.split(';')[1].split()[0]
        try:
            if ii == 0:
                # 不需要mask
                x = Image.open(annotation_path)
            else:
                x = add_mask(annotation_path, mask)

            label = int(line.split(';')[0])
            # 判断胸片位置
            class_name_late = classfication_front.detect_image(x)

            class_name, probability, pred, class_id = classfication.detect_image(x)

            tabel_write_list = [annotation_path, label, class_id]
            tabel_write_list += pred.tolist()

            # 'Frontal' = 0
            if class_name_late[-1] == 0 or ii == 0:
                csv_writer.writerow(
                    tabel_write_list)
            # Lateral == 1
            elif class_name_late[-1] == 1 and ii != 0:
                csv_lateral.writerow(
                    tabel_write_list)
        except:
            print('Error: cant not load image file :', annotation_path)

    # 5. 关闭文件
    f.close()
    f_lateral.close()


def eva_roc(metrics_out_path, use_95_confiden=False):
    labels = class_names

    txt_line = open(metrics_out_path + '/eval_result.txt', 'w')
    csv_path = metrics_out_path + '/roc_eval.csv'

    data = pd.read_csv(csv_path)
    print(data.head())
    true_y = data['y_real'].to_numpy()
    true_y = to_categorical(true_y, num_classes=len(labels))
    PM_y = data[labels].to_numpy()
    n_classes = PM_y.shape[1]
    print(n_classes, '- > n_classes\n')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    thresholds = dict()
    for i in range(n_classes):
        # print('ground_truth -> ', true_y[:, i], 'y_prediction -> ', PM_y[:, i])

        if use_95_confiden:
            # 求95%置信区间AUC
            try:
                confidence_interval = conf_auc(PM_y[:, i], true_y[:, i])
            except:
                confidence_interval = 0
        else:
            confidence_interval = None

        fpr[i], tpr[i], _ = roc_curve(true_y[:, i], PM_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        print(labels[i], ' -> ', roc_auc[i])
        txt_line.write(str(labels[i]) + ' -> ' + str(roc_auc[i]))
        txt_line.write('\n')
        txt_line.write(str(confidence_interval))
        txt_line.write('\n')

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # print('\nall_fpr ->', all_fpr)
    # print('mean_tpr ->', mean_tpr)
    print('macro-average AUC ->', roc_auc["macro"])
    txt_line.write('macro-average AUC ->' + str(roc_auc["macro"]))
    txt_line.close()

    lw = 2
    # Plot all ROC curves
    plt.figure()

    if math.isnan(roc_auc["macro"]) == False:
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.4f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

    import matplotlib.colors as mcolors
    colors = cycle(list(mcolors.TABLEAU_COLORS.keys()))  # 颜色变化
    # colors=cycle(list(mcolors.CSS4_COLORS.keys()) )#颜色变化

    for i, color in zip(range(n_classes), colors):
        if math.isnan(roc_auc[i]) == False:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=labels[i] + '(area = {0:0.4f})'.format(roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity (%)')
    plt.ylabel('Sensitivity (%)')
    plt.title('Some extension of Receiver operating characteristic to multi-class')

    def to_percent(temp, position):
        return '%1.0f' % (100 * temp)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.legend(loc="best")

    # plt.savefig(os.path.join(metrics_out_path, "Roc_eval.png"))
    # 自动保存最大化格式
    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(os.path.join(metrics_out_path, "Roc_eval.png"), bbox_inches='tight', dpi=300)

    plt.savefig(os.path.join(metrics_out_path, "Roc_eval.pdf"))
    # plt.show()
    plt.close()


def conf_auc(test_predictions, ground_truth, bootstrap=1000, seed=None, confint=0.95):
    import numpy as np
    import sklearn
    from sklearn import metrics
    """Takes as input test predictions, ground truth, number of bootstraps, seed, and confidence interval"""
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    if confint > 1:
        confint = confint / 100
    for i in range(bootstrap):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(test_predictions) - 1, len(test_predictions))
        if len(np.unique(ground_truth[indices])) < 2:
            continue

        score = metrics.roc_auc_score(ground_truth[indices], test_predictions[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound = (1 - confint) / 2
    upper_bound = 1 - lower_bound
    confidence_lower = sorted_scores[int(lower_bound * len(sorted_scores))]
    confidence_upper = sorted_scores[int(upper_bound * len(sorted_scores))]
    auc = metrics.roc_auc_score(ground_truth, test_predictions)
    print(
        "{:0.0f}% confidence interval for the score: [{:0.6f} - {:0.6}] and your AUC is: {:0.6f}".format(confint * 100,
                                                                                                         confidence_lower,
                                                                                                         confidence_upper,
                                                                                                         auc))
    confidence_interval = (confidence_lower, auc, confidence_upper)
    return confidence_interval


if __name__ == '__main__':

    # 获取类名
    classes_path = 'model_data/cls_classes.txt'
    class_names, num_classes = get_classes(classes_path)

    test_annotation_path = 'data_path_txt/10.9/test_path_test+vinder+pcxr_last2.txt'

    ii: int = 0
    # root_path = 'Mask_auc_out_vindr-pcxr'
    root_path = 'mask_ouput/Mask_auc_out_test_10.6'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    lateral_path = root_path + '/Lateral'
    if not os.path.exists(lateral_path):
        os.makedirs(lateral_path)
    # 无mask auc评估
    metrics_out_path = root_path + "/mask" + str(ii)
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
    write_csv(metrics_out_path, test_annotation_path, root_path, ii, mask=None)
    eva_roc(metrics_out_path, True)
    ii += 1
    # mask auc评估
    for mask in mask_list:
        metrics_out_path = root_path + "/mask" + str(ii)
        if not os.path.exists(metrics_out_path):
            os.makedirs(metrics_out_path)
        write_csv(metrics_out_path, test_annotation_path, root_path, ii, mask)
        eva_roc(metrics_out_path, True)
        ii += 1
    # 侧位auc评估
    eva_roc(lateral_path, True)
