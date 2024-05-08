# Author:司震
# Create: 2024/4/20
# Description: 第12天练习：数据降维
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#利用PCA（主成分分析法），选择可解释方差比对鸢尾花数据进行降维
# X,y=load_iris(return_X_y=True)#生成数据集
# pca = PCA(n_components=0.95)#生成PCA转换器,降到可解释方差比总和为95%的维度
# X_reduced=pca.fit_transform(X,y)#拟合及转换原始数据
# print(X_reduced.shape)
# plt.figure(dpi=200)
#将转换后的数据表示出来
# plt.scatter(X_reduced[y==0,0],X_reduced[y==0,1])
# plt.scatter(X_reduced[y==1,0],X_reduced[y==1,1])
# plt.scatter(X_reduced[y==2,0],X_reduced[y==2,1])
# plt.legend(['setosa','versicolor','virginica'])
#plt.savefig('pca.png')
# plt.show()






