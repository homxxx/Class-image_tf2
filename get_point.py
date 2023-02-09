'''
获取图片鼠标坐标
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("lung_mask_exp.png")
image = cv2.resize(image, (512, 512))
print('image.shape[:2] ->',image.shape[:2])
plt.imshow(image)
pos=plt.ginput(8)
print(pos)
# plt.show()