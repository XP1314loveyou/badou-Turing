'''
密度聚类
这个数据集里一共包括150行记录，其中前四列为花萼长度，花萼宽度，花瓣长度，花瓣宽度等4个
用于识别鸢尾花的属性，第5列为鸢尾花的类别（包括Setosa，Versicolour，Virginica三类）
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from  sklearn.cluster import DBSCAN
#获取数据集,该数据集有data（数据集）和target（数据标签）两个属性
iris=datasets.load_iris()
print(iris)
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X)
#密度聚类，默认metric='euclidean'：采用欧氏距离计算
dbscan=DBSCAN(eps=0.4,#半径
              min_samples=9)#构成簇的最少点数

dbscan.fit(X)
#聚类结果，如果是-1，说明算法认为这个点是噪声点
label_pred = dbscan.labels_

# 绘制结果
x0 = X[label_pred == 0]#掩码数组
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)#可以是1,2,3,4，第二象限为左上角
plt.show()
