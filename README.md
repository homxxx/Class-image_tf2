## Tensorflow2 图像分类模型
---


## 所需环境

scipy==1.4.1
numpy==1.18.4
matplotlib==3.2.1
opencv_python==4.2.0.34
tensorflow_gpu==2.2.0
tqdm==4.46.1
Pillow==8.2.0
h5py==2.10.0

## 热力图
Mobile net：

<img src="https://github.com/homxxx/Class-image_tf2/blob/master/doc/mobilenet-75-conv_pw_12_relu.png" width="300px">

Resnet：

<img src="https://github.com/homxxx/Class-image_tf2/blob/master/doc/resnet-172-add_16.png" width="300px">


## 训练步骤
1. datasets文件夹可以存放的图片分为两部分，train里面是训练图片，test里面是测试图片。  
2. 在训练之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。文件格式可参考如下：
```
|-datasets
    |-train
        |-label1
            |-123.jpg
            |-234.jpg
        |-label2
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-label1
            |-567.jpg
            |-678.jpg
        |-label2
            |-789.jpg
            |-890.jpg
        |-...
```
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的txt路径文件，运行前需要修改其中的classes，将其修改成自己需要分的类。   
4. 之后修改model_data文件夹下的cls_classes.txt，使其也对应自己需要分的类。  
5. 在train.py里面调整参数开始训练。
   
6. 也可以通过标签txt构建绝对路径训练数据集txt文件索引，标签代号与标签txt文件对应即可。根据不同数据集的标注格式来构建全局数据集路径。具体代码参考data_pre文件夹。

## 预测步骤
1. 按照训练步骤训练。  
2. 在classification.py文件里面，在如下部分修改model_path、classes_path、backbone和alpha使其对应训练好的文件；
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/model.h5',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   输入的图片大小
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   所用模型种类：
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #--------------------------------------------------------------------#
    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    #--------------------------------------------------------------------#
    "alpha"         : 0.25
}
```
3. 运行predict.py，输入图像路径  
```python
img/test.jpg
```  


## 评估步骤

1. 在准备好数据集后，需要在运行txt_annotation.py或者根据csv标签构建评估所需的数据路径文件test_annotation_path.txt。  
2. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
**_示例：_**
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/model.h5',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   输入的图片大小
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   所用模型种类：
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #--------------------------------------------------------------------#
    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    #--------------------------------------------------------------------#
    "alpha"         : 0.25
}
```
3. 运行eval_auc_95-confiden.py来进行模型AUC ROC准确率评估。
   eval_auc_95-confiden.py运行前需要修改的参数：
   ```
    classes_path ： 模型标签 
    test_annotation_path ： 测试集路径文件
    root_path ： 结果输出保存路径
    ```

## 模型热力图获得步骤
1. 在准备好数据集后，需要在运行txt_annotation.py或者根据csv标签构建评估所需的数据路径文件test_annotation_path.txt。
2. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。
3. 运行heatmap-new_version.py来进行模型热力图结果。
   heatmap-new_version.py运行前需要修改的参数：

   示例：
   ```python
    # 数据集路径path文件
    test_annotation_path = 'data_path_txt/data_path_vindr-pcxr_train.txt'
    # 获取类名
    classes_path = 'model_data/cls_classes_new2.txt'
    # 结果输出保存路径
    output_dest = 'heatmap_output/vindr-pcxr/'
    ```


## Mask分区贡献率评估步骤

1. 在准备好数据集后，需要在运行txt_annotation.py或者根据csv标签构建评估所需的数据路径文件test_annotation_path.txt。
2. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
   **_示例：_**
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/model.h5',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   输入的图片大小
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   所用模型种类：
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #--------------------------------------------------------------------#
    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    #--------------------------------------------------------------------#
    "alpha"         : 0.25
}
```
3. 运行eval_auc_mask-95-confiden.py来进行模型mask贡献率准确率评估。
   eval_auc_mask-95-confiden.py运行前需要修改的参数：
   ```
    classes_path ： 模型标签 
    test_annotation_path ： 测试集路径文件
    root_path ： 结果输出保存路径
    ```
   mask遮罩坐标数据与处理函数在add_mask.py。遮罩具体参考lung_mask_exp.png文件。
   
   获得遮罩数据代码：get_point.py
   
    判断正侧位模型使用mobilenet，遮罩评估的时候为级联结构，模型封装代码classification_front.py


## 网络结构图
运行plot_model.py

## 与人类结果对比 绘图
compare_human_plot.py
修改参数：
示例：
   ```python
    机器结果
    csv_path = 'compare_with_human/metrics_out_human/machine.csv'
    # 获取类名
    classes_path = 'model_data/cls_classes_new2_delete_last2.txt'
    
    root_path = 'compare_with_human/com_plot'
    ```
