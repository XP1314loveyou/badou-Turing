"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]#X列表
print(X)

cluster_model=KMeans(n_clusters=3)
print('cluster_model:',cluster_model)

y_pred=cluster_model.fit_predict(X)
print('y_pred:',y_pred)#一维数组，为类别标记
#分别取X的第一列和第二列
x=[n[0] for n in X]
y=[n[1] for n in X]

plt.scatter(x,y,c=y_pred,marker='x')#c即color,可以是颜色或者颜色序列

#绘制标题
plt.title("Kmeans-Cluster-Graph")

plt.rcParams['font.sans-serif'] = ['SimHei']
#绘制x轴和y轴坐标
plt.xlabel('每分钟助攻数')
plt.ylabel('每分钟得分数')
# 设置右上角图例
plt.legend(labels=["A", "B", "C"],loc='best')

# 显示图形
plt.show()