'''
灰度图像直方图均衡化
hist=cv2.calcHist([image1,image2,...],
                [0,0,...],
                None,
                [256,256,...],
                [0,255,0,255,...])
np.hstack():沿着水平方向将数组堆叠起来。
np.hstack(())和np.hstack([])的区别
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png',1)#1 ：彩色  0 ：灰色
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
dst=cv2.equalizeHist(gray_img)#均衡化的图像，非直方图
# hist是一个shape为(256,1)的数组，表示0-255每个像素值对应的像素个数，下标即为相应的像素值
hist=cv2.calcHist([dst]#图像
                  ,[0]#通道数
                  ,None#掩膜，是一个大小和image一样的np数组，需要处理的部分指定为1，其余为0，一般设置为None，表示处理整幅图像
                  ,[256],#使用多少柱子
                  [0,255])#像素值范围
plt.figure()
plt.plot(hist)
plt.hist(dst.ravel(),#多维数组需要拉生成一维
         256)#柱数,即默认x轴从0-255
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst]))
cv2.waitKey()
cv2.destroyAllWindows()























