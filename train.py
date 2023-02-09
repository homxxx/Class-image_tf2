import datetime
import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets import freeze_layers, get_model_from_name
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.dataloader import ClsDatasets
from utils.utils import get_classes, get_lr_scheduler
from utils.utils_fit import fit_one_epoch


def read_txt(in_txt_list):
    res = []
    for in_txt in in_txt_list:
        with open(in_txt, encoding='utf-8') as f:
            lines_temp = f.readlines()
        res += lines_temp
    return res


if __name__ == "__main__":
    #   是否使用eager模式训练
    eager = False

    #   train_gpu   训练用到的GPU
    #               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    train_gpu = [0, 1]

    #   classes_path    模型的label文件
    classes_path = 'model_data/cls_classes.txt'

    #   input_shape     输入的shape大小
    input_shape = [512, 512]

    #   所用模型种类：
    #   mobilenet、resnet50、vgg16、vit; resnet50_sigmoid
    backbone = "resnet50"

    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    alpha = 0.25

    #   预训练权值 ： 当model_path = ''的时候不加载整个模型的权值。
    model_path = ""

    Init_Epoch = 5
    Freeze_Epoch = 1
    Freeze_batch_size = 32

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 16

    Freeze_Train = False

    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #   momentum        优化器内部使用到的momentum参数
    # ------------------------------------------------------------------#

    # SGD
    Init_lr = 1e-2

    Min_lr = Init_lr * 0.01

    # optimizer_type = "adam"
    optimizer_type = "sgd"
    momentum = 0.9
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    # lr_decay_type = 'cos'
    lr_decay_type = 'step'
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    # all dataset
    save_dir = 'checkpoint'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # ------------------------------------------------------------------#
    #   num_workers     多线程
    #   use_multi_processing 多进程
    # ------------------------------------------------------------------#
    num_workers = 2
    use_multi_processing = False
    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   test_annotation_path    验证图片路径和标签
    #   train_list test_list 路径列表
    # ------------------------------------------------------#

    train_annotation_path = "data_path_txt/6.21/cls_train_all-621.txt"
    train_annotation_path_2 = "cls_test.txt"

    test_annotation_path = 'data_path_txt/6.21/cls_test_all-621.txt'
    # brax
    train_annotation_path_brax = "data_path_txt/data_path_brax.txt"
    # vindr
    train_annotation_path_vindr = "data_path_txt/data_path_vindr_train.txt"
    test_annotation_path_vindr = 'data_path_txt/data_path_vindr_test.txt'
    # vindr-pcxr
    train_annotation_path_vindr_pcxr = "data_path_txt/data_path_vindr-pcxr_train.txt"
    test_annotation_path_vindr_pcxr = 'data_path_txt/data_path_vindr-pcxr_test.txt'


    train_list: list = [train_annotation_path, train_annotation_path_2, train_annotation_path_brax,
                        train_annotation_path_vindr, train_annotation_path_vindr_pcxr,
                        ]
    test_list: list = [test_annotation_path_vindr, test_annotation_path_vindr_pcxr, test_annotation_path]

    if backbone == "resnet50_sigmoid":
        loss_f = 'binary_crossentropy'
    else: loss_f = 'categorical_crossentropy'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))

    class_names, num_classes = get_classes(classes_path)

    if ngpus_per_node > 1:
        with strategy.scope():
            if backbone == "mobilenet":
                model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3],
                                                      classes=num_classes, alpha=alpha)
            else:
                model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3],
                                                      classes=num_classes)
            if model_path != "":
                print('Load weights {}.'.format(model_path))
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:
        if backbone == "mobilenet":
            model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes,
                                                  alpha=alpha)
        else:
            model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes)
        if model_path != "":
            print('Load weights {}.'.format(model_path))
            model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    # 是否直接在训练集中选取测试集
    Small_Train = False
    if Small_Train:
        print('Small_Training ...')
        val_split = 0.1

        with open(train_annotation_path, "r") as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

        num_val_fromtrain = int(len(lines) * val_split)
        num_train = len(lines) - num_val_fromtrain

        train_lines = lines[:num_train]
        val_lines = lines[num_train:]

        num_val = num_val_fromtrain

    else:
        print('all_training ...')
        train_lines = read_txt(train_list)
        val_lines = read_txt(test_list)
        num_train = len(train_lines)
        num_val = len(val_lines)

        np.random.seed(10101)
        np.random.shuffle(train_lines)
        np.random.seed(None)

    if True:
        if Freeze_Train:
            freeze_layers = freeze_layers[backbone]
            for i in range(freeze_layers): model.layers[i].trainable = False
            # print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))
            print('Freeze_Train={}\nFreeze the first {} layers of total {} layers.'.format(Freeze_Train, freeze_layers,
                                                                                           len(model.layers)))

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == 'vit':
            nbs = 256
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader = ClsDatasets(train_lines, input_shape, batch_size, num_classes, train=True)
        val_dataloader = ClsDatasets(val_lines, input_shape, batch_size, num_classes, train=False)

        optimizer = {
            # adam+sgd : amsgrad = True
            'adam': Adam(lr=Init_lr_fit, beta_1=momentum, amsgrad=True),
            'sgd': SGD(lr=Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]

        if eager:
            start_epoch = Init_Epoch
            end_epoch = UnFreeze_Epoch
            UnFreeze_flag = False

            gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
            gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history = LossHistory(log_dir)

            for epoch in range(start_epoch, end_epoch):

                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size = Unfreeze_batch_size

                    nbs = 64
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                    if backbone == 'vit':
                        nbs = 256
                        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                        lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    for i in range(len(model.layers)):
                        model.layers[i].trainable = True

                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    train_dataloader.batch_size = batch_size
                    val_dataloader.batch_size = batch_size

                    gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

                    gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
                    gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

                    UnFreeze_flag = True

                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)

                fit_one_epoch(model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                              end_epoch, save_period, save_dir, strategy)

                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()
        else:
            start_epoch = Init_Epoch
            end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch

            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(loss=loss_f, optimizer=optimizer,
                                  metrics=['categorical_accuracy'])
            else:
                model.compile(loss=loss_f, optimizer=optimizer, metrics=['categorical_accuracy'])

            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(save_dir, "loss_" + str(time_str))
            logging = TensorBoard(log_dir)
            loss_history = LossHistory(log_dir)
            checkpoint = ModelCheckpoint(
                os.path.join(save_dir, str(backbone) + '-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-acc{'
                                                       'categorical_accuracy:.3f}-val_acc{'
                                                       'val_categorical_accuracy:.3f}.h5'),
                monitor='val_loss', save_weights_only=True, save_best_only=False, period=save_period)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1)
            lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
            callbacks = [logging, loss_history, checkpoint, lr_scheduler]

            if start_epoch < end_epoch:
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                           batch_size))
                print('input_shape -> {} ; backbone -> {} '.format(input_shape, backbone))

                model.fit(
                    x=train_dataloader,
                    steps_per_epoch=epoch_step,
                    validation_data=val_dataloader,
                    validation_steps=epoch_step_val,
                    epochs=end_epoch,
                    initial_epoch=start_epoch,
                    # use_multiprocessing=True if num_workers > 1 else False,
                    use_multiprocessing=use_multi_processing,
                    workers=num_workers,
                    callbacks=callbacks
                )
            if Freeze_Train:
                batch_size = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch = UnFreeze_Epoch

                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == 'vit':
                    nbs = 256
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
                callbacks = [logging, loss_history, checkpoint, lr_scheduler]

                for i in range(len(model.layers)):
                    model.layers[i].trainable = True
                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(loss=loss_f, optimizer=optimizer,
                                      metrics=['categorical_accuracy'])
                else:
                    model.compile(loss=loss_f, optimizer=optimizer,
                                  metrics=['categorical_accuracy'])

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                train_dataloader.batch_size = Unfreeze_batch_size
                val_dataloader.batch_size = Unfreeze_batch_size

                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                           batch_size))
                print('input_shape -> {} ; backbone -> {} '.format(input_shape, backbone))
                model.fit(
                    x=train_dataloader,
                    steps_per_epoch=epoch_step,
                    validation_data=val_dataloader,
                    validation_steps=epoch_step_val,
                    epochs=end_epoch,
                    initial_epoch=start_epoch,
                    # use_multiprocessing=True if num_workers > 1 else False,
                    use_multiprocessing=use_multi_processing,
                    workers=num_workers,
                    callbacks=callbacks
                )
