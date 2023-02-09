from .mobilenet import MobileNet
from .resnet50 import ResNet50
from .vgg16 import VGG16
from .vit import VisionTransformer
from .resnet50_sigmoid import ResNet50_sigmoid

get_model_from_name = {
    "mobilenet"     : MobileNet,
    "resnet50"      : ResNet50,
    "vgg16"         : VGG16,
    "vit"           : VisionTransformer,
    "resnet50_sigmoid"      : ResNet50_sigmoid,
}

freeze_layers = {
    "mobilenet"     : 81,
    "resnet50"      : 173,
    "vgg16"         : 19,
    "vit"           : 130,
    "resnet50_sigmoid"      : 173,
}
