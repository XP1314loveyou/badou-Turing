'''
彩色图像直方图均衡化
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 1)
# 分解通道
b, g, r = cv2.split(img)
b_channel = cv2.equalizeHist(b)
g_channel = cv2.equalizeHist(g)
r_channel = cv2.equalizeHist(r)

new_img = cv2.merge((b_channel, g_channel, r_channel))
cv2.imshow('new_img', new_img)#均衡化图像
cv2.imshow('img', img)#原彩色图
cv2.imshow('red', r)#红色通道图,灰度图

cv2.waitKey()
cv2.destroyAllWindows()
