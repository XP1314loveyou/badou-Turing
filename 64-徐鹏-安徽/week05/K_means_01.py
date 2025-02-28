'''
OpenCV中，Kmeans()函数:
retval, bestLabels, centers = kmeans(
    data, 聚类数据，最好是np.float32类型
    K,    聚类类簇数
    bestLabels,输出的整数数组，用于存储每个样本的类别标签索引如0,1,2
    criteria, 迭代停止的模式选择，格式为（type, max_iter, epsilon），type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts, 重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags[, centers])  表示初始中心的选择，两种方法：
                        cv2.KMEANS_PP_CENTERS ：使用kmeans++算法的中心初始化算法，即初始中心的选择使眼色相差最大
                        cv2.KMEANS_RANDOM_CENTERS：每次随机选择初始中心
retval:紧密度，返回每个点到相应中心的距离的平方和
centers:由聚类的中心组成的数组,每个集群中心为一行数据

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取彩色图像的灰度图像
img = cv2.imread('lenna.png', 0)
print(img)
print(img.shape)

#获取图像高度、宽度
rows,cols=img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))#拉成一列(还是二维),拉成一维用ravel或flatten注意二者区别
data = np.float32(data)
print('data shape:',data.shape)
print('data:',data)
#设置迭代停止条件：迭代10次，误差为1
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness,labels,centers=cv2.kmeans(data,4,None,criteria,10,flags)
print('label shape:',labels.shape)
print('labels:',labels)
#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
