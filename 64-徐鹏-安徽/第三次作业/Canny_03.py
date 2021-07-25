'''
Canny边缘检测：设置调节杆，用于优化
'''
import numpy as np
import cv2

# 回调函数
def CannyThreshold(lowThreshold):
    detected_edges=cv2.GaussianBlur(gray#输入图像
                                    ,(3,3)#高斯核大小
                                    ,0)#x方向标准差
    detected_edges = cv2.Canny(detected_edges,#要检测的图像
                               lowThreshold,#最小值阈值
                               lowThreshold*ratio,#最大值阈值
                               apertureSize=kernel_size)#sobel算子（卷积核）大小

    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 用原始颜色添加到检测的边缘上
    cv2.imshow('canny demo', dst)

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图

cv2.namedWindow('canny demo')#图像框可以自行调整大小

#设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
cv2.createTrackbar('Minist Threshold',
                   'canny demo',
                   lowThreshold,
                   max_lowThreshold,
                   CannyThreshold)

CannyThreshold(0)#初始情况
if cv2.waitKey(0)==27:#按esc键退出
    cv2.destroyAllWindows()
'''
cv2.waitkey（delay）函数
1.若参数delay≤0：表示一直等待按键；
2、若delay取正整数：表示等待按键的时间，比如cv2.waitKey(30)，就是等待30（milliseconds）；（视频中一帧数据显示（停留）的时间）

cv2.waitKey(delay)返回值：
1、等待期间有按键：返回按键的ASCII码（比如：Esc的ASCII码为27，即0001  1011）；
2、等待期间没有按键：返回 -1；
'''




