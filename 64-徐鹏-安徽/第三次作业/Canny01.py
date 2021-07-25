'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1,低于阈值1的像素点会被认为不是边缘；
第三个参数是滞后阈值2,高于阈值2的像素点会被认为是边缘;
在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘
'''
import cv2

img=cv2.imread('lenna.png',1)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny_img=cv2.Canny(img_gray,200,300)
cv2.imshow('canny01',canny_img)

cv2.waitKey()
cv2.destroyAllWindows()


















