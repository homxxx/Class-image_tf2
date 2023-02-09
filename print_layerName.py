'''
@homxxx
Keras获取最后一层的名字
Keras获取所有层的名字
'''

# import keras
from classification import Classification

classfication = Classification()
model = classfication.model
# print(model.summary())
model.summary()

# layer_name = None
for i, layer in enumerate(model.layers):
    # print(i, layer.name, (layer))
    print(i, layer.name)
    # print(i, layer.name, layer.type)
    layer_name = layer.name
#
# print('最后一个卷积层的名字:', layer_name)
