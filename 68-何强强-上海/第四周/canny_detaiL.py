# -*- coding:utf-8 -*-
"""
Canny是目前最优秀的边缘检测算法，其目标为找到一个最优的边缘，其最优边缘的定义为：
    1、好的检测：算法能够尽可能的标出图像中的实际边缘
    2、好的定位：标识出的边缘要与实际图像中的边缘尽可能接近
    3、最小响应：图像中的边缘只能标记一次
实现步骤：
    1. 对图像进行灰度化
    2. 对图像进行高斯滤波： 根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样 可以有效滤去理想图像中叠加的高频噪声。
    3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
    4 对梯度幅值进行非极大值抑制
    5 用双阈值算法检测和连接边缘
"""

