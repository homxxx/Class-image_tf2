
import tensorflow as tf
from PIL import Image

from classification import Classification

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

classfication = Classification()


img = r''

image = Image.open(img)

class_name = classfication.detect_image(image)
print(class_name)


