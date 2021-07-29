'''
层次聚类
linkage(y, method='single', metric='euclidean', optimal_ordering=False)
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法，可以是single,complete,average,weighted,centroid,median,ward等方法
3. metric：str或function，可选。在y维观测向量的情况下使用该参数，否则忽略。
参照有效距离度量列表的pdist函数，还可以使用自定义距离函数。

4. optimal_ordering:bool。若为true，linkage矩阵则被重新排序，以便连续叶子间距最小。
当数据可视化时，这将使得树结构更为直观。默认为false，因为数据集非常大时，执行此操作计算量将非常大。

fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None)
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息;
2.t是一个聚类的阈值
3.criterion: str,可选参数。形成扁平簇的准则，可以是：inconsistent，distance，maxclust，monocrit
4.depth：int, 可选参数。执行不一致计算的最大深度。对于其他标准没有任何意义。默认值为2
5.R：ndarray, 可选参数。用于‘inconsistent’准则的不一致矩阵。如果未提供，则计算该矩阵。
6.monocrit：ndarray, 可选参数。
'''
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
#给定数据点
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
#层次聚类，返回的结果是一个矩阵，是由聚类结果编码而成的
Z=linkage(X,method='ward')
print(Z)
# 从给定链接矩阵定义的层次聚类中形成平面聚类
f=fcluster(Z,4,'distance')
#设置画布
fig = plt.figure(figsize=(5, 3))
#画树状图
dn = dendrogram(Z)
plt.show()
