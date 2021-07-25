'''
利用接口实现PCA
PCA(copy=True,需要将原始训练数据复制
 n_components=2,保留维度个数
 whiten=False)使得每个特征具有相同的方差
'''

import numpy as np
from sklearn.decomposition import PCA

X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4

pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
print(pca.explained_variance_ratio_)#输出贡献率，即所保留各个特征的方差百分比
print(newX)#输出降维后的数据
