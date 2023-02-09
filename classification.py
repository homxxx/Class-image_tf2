import os

import matplotlib.pyplot as plt
import numpy as np

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)


class Classification(object):
    _defaults = {
        # 模型路径
        "model_path": '',

        # 模型label
        "classes_path": 'model_data/cls_classes.txt',

        "input_shape": [512, 512],

        #   mobilenet、resnet50、vgg16、vit 、 resnet50_sigmoid
        "backbone": 'resnet50_sigmoid',

        #   仅在backbone='mobilenet'的时候有效
        "alpha": 0.25,

        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        if self.backbone == "mobilenet":
            self.model = get_model_from_name[self.backbone](input_shape=[self.input_shape[0], self.input_shape[1], 3],
                                                            classes=self.num_classes, alpha=self.alpha)
        else:
            self.model = get_model_from_name[self.backbone](input_shape=[self.input_shape[0], self.input_shape[1], 3],
                                                            classes=self.num_classes)
        self.model.load_weights(self.model_path)
        print('{} model loaded\n classes {}\n num_classes{}.'.format(model_path, self.class_names, self.num_classes))

    def detect_image(self, image):
        image = cvtColor(image)

        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        preds = self.model.predict(image_data)[0]

        class_id = np.argmax(preds)
        class_name = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        return class_name, probability, preds, class_id
