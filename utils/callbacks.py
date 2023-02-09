import os
import warnings
import math

from tensorflow import keras
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import backend as K


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []

        os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        self.train_acc.append(logs.get('categorical_accuracy'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_train_accuracy.txt"), 'a') as f:
            f.write(str(logs.get('categorical_accuracy')))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_accuracy.txt"), 'a') as f:
            f.write(str(logs.get('val_categorical_accuracy')))
            f.write("\n")

        self.loss_plot()
        self.loss_plot_m()
        self.acc_plot_m()
        self.train_acc_plot()
        self.val_acc_plot()
        # print time
        import time
        localtime = time.asctime(time.localtime(time.time()))
        print("本地时间 :", localtime)

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def loss_plot_m(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        # plt.plot(iters, self.train_acc, 'b--', linewidth = 2, label='train_acc')
        # plt.plot(iters, self.val_acc, 'c--', linewidth = 2, label='val_acc')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')

            # plt.plot(iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'b--', linestyle = '--', linewidth = 2, label='smooth train acc')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), 'c--', linestyle = '--', linewidth = 2, label='smooth val acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss_.png"))
        # 矢量图
        plt.savefig(os.path.join(self.log_dir, "epoch_loss_.svg"), format="svg")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss_.pdf"))

        plt.cla()
        plt.close("all")

    def acc_plot_m(self):
        '''
        记录acc
        :return:
        '''
        iters = range(len(self.losses))

        plt.figure()
        # plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        # plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        plt.plot(iters, self.train_acc, 'b', linewidth=2, label='train_acc')
        plt.plot(iters, self.val_acc, 'c', linewidth=2, label='val_acc')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            # plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
            #          label='smooth train loss')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
            #          linewidth=2, label='smooth val loss')

            # plt.plot(iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'b--', linestyle='--', linewidth=2,
            plt.plot(iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'b', linestyle='--', linewidth=2,
                     label='smooth train acc')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), 'c--', linestyle='--', linewidth=2,
            plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), 'c', linestyle='--', linewidth=2,
                     label='smooth val acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.title('A Acc Curve')
        # plt.legend(loc="upper right")
        plt.legend(loc="best")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc_all.png"))
        # 矢量图
        plt.savefig(os.path.join(self.log_dir, "epoch_acc_all.svg"), format="svg")
        plt.savefig(os.path.join(self.log_dir, "epoch_acc_all.pdf"))

        plt.cla()
        plt.close("all")

    def train_acc_plot(self):
        '''
        记录acc
        :return:
        '''
        iters = range(len(self.losses))

        plt.figure()
        # plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        # plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        plt.plot(iters, self.train_acc, 'b', linewidth=2, label='train_acc')
        # plt.plot(iters, self.val_acc, 'c', linewidth=2, label='val_acc')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'b--', linestyle='--', linewidth=2,
        #              label='smooth train acc')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), 'c--', linestyle='--', linewidth=2,
        #              label='smooth val acc')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.title('Train Acc Curve')
        plt.legend(loc="best")

        plt.savefig(os.path.join(self.log_dir, "epoch_train_acc.png"))
        # 矢量图
        plt.savefig(os.path.join(self.log_dir, "epoch_train_acc.svg"), format="svg")
        plt.savefig(os.path.join(self.log_dir, "epoch_train_acc.pdf"))

        plt.cla()
        plt.close("all")

    def val_acc_plot(self):
        '''
        记录acc
        :return:
        '''
        iters = range(len(self.losses))

        plt.figure()
        # plt.plot(iters, self.train_acc, 'b', linewidth=2, label='train_acc')
        plt.plot(iters, self.val_acc, 'c', linewidth=2, label='val_acc')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.title('A Val Acc Curve')
        plt.legend(loc="best")

        plt.savefig(os.path.join(self.log_dir, "epoch_val_acc.png"))
        # 矢量图
        plt.savefig(os.path.join(self.log_dir, "epoch_val_acc.svg"), format="svg")
        plt.savefig(os.path.join(self.log_dir, "epoch_val_acc.pdf"))

        plt.cla()
        plt.close("all")


class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate = decay_rate
        self.verbose = verbose
        self.learning_rates = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.init_lr = 0
        self.last_epoch = 0

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, batch, logs=None):
        learning_rate = self.eta_min + (self.init_lr - self.eta_min) * (
                    1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        self.last_epoch += 1

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
