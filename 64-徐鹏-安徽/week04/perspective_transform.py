'''
cv2.getPerspectiveTransform(src, #原图像即数组（图像坐标点）
                            dst) #目标图像即数组（图像坐标点）

cv2.warpPerspective(src, 输入图像
                    M,  变换矩阵
                    dsize,变换后输出图像尺寸
                    dst=None,输出图像，默认None
                    flags=None,插值方法
                    borderMode=None,边界像素外扩方式
                    borderValue=None)边界像素插值，默认用0填充
'''


import numpy as np
import cv2

img=cv2.imread('photo1.jpg')
img1=img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

print(img.shape)

# 生成透视变换矩阵；进行透视变换
mt=cv2.getPerspectiveTransform(src,dst)

print('透视变换矩阵:')
print(mt)
#透视变换
result=cv2.warpPerspective(img1,#需变换的图像
                           mt,#变换矩阵
                           (338,488))#变换后图像大小

cv2.imshow('original image:',img)
cv2.imshow('transformed image:',result)

cv2.waitKey()
cv2.destroyAllWindows()
