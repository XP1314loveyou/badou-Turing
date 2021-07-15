'''
灰度图像及彩色图像直方图
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：[0]
mask：掩膜，一般为：None
histSize：直方图柱数，一般等于灰度级数256 ：[256]
ranges：横轴范围:[0,255]
'''
# 灰度图像的直方图，方法一
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.title("Gray_Histogram")
plt.xlabel("x")#X轴标签
plt.ylabel("y")#Y轴标签
plt.hist(gray.ravel(), 256)
plt.show()

# 灰度图像的直方图, 方法二

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Gray_Histogram1")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,255])#设置x坐标轴范围
plt.show()

#彩色图像直方图
image = cv2.imread("lenna.png")#默认读取彩色图像
cv2.imshow("Original",image)

channels=cv2.split(image)
colors=('b','g','r')

plt.figure()
plt.title('Flattened Color Histogram')
plt.xlabel('x')
plt.ylabel('y')

for chan,color in zip(channels,colors):
    hist=cv2.calcHist([chan],[0],None,[256],[0,255])
    plt.plot(hist,color=color)
    plt.xlim([0,255])
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()















