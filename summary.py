#--------------------------------------------#
#   该部分代码只用于看网络结构
#--------------------------------------------#
from nets.mobilenet import MobileNet
from nets.resnet50 import ResNet50
from nets.vgg16 import VGG16
from nets.vit import VisionTransformer

if __name__ == "__main__":
    model = ResNet50([512, 512, 3], classes=26)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
