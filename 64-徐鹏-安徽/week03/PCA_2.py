import numpy as np

class PCA():
    def __init__(self,dimension_para):
        self.dimension_para=dimension_para
    def dimension_transform(self,X):
        self.features=X.shape[1]
        # 求协方差矩阵(先中心化)
        X=X-X.mean(axis=0)
        self.covariance=np.dot(X.T,X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors=np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号,argsort :默认获取从小到大排序的索引值
        indexs=np.argsort(-eigenvalues)
        # 降维矩阵
        self.components_=eigenvectors[:,indexs[:self.dimension_para]]
        # 对X进行降维
        return np.dot(X,self.components_)
# 调用
pca = PCA(dimension_para=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX=pca.dimension_transform(X)
print(newX)


































































