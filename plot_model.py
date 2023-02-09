import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
from nets.mobilenet import MobileNet
from nets.resnet50 import ResNet50
from nets.vgg16 import VGG16
from nets.vit import VisionTransformer
from nets.resnet50_sigmoid import ResNet50_sigmoid
from keras.utils import plot_model
# from keras.utils.vis_utils import plot_model
# import graphviz

if __name__ == "__main__":
    model = ResNet50_sigmoid([512, 512, 3], classes=26)
    # model.summary()
    #
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)

    plot_model(model, to_file='model-ResNet50_sigmoid.png', show_shapes=True, expand_nested=True)
    print('done')