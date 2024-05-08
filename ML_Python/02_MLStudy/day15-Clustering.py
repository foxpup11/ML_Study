# Author:司震
# Create: 2024/4/18
# Description: 第15天练习：聚类
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

#K-means聚类
# X1,y1=datasets.make_circles(n_samples=500,factor=.6,noise=.05)#生成环形数据集
# X2,y2=datasets.make_blobs(n_samples=100,n_features=2,centers=[[1.2,1.2]],cluster_std=[[.1]],random_state=9)#生成斑点状数据集
# X =np.concatenate((X1,X2))#连接
# y_pred = KMeans(n_clusters=3,random_state=9).fit_predict(X)
# #将类别划分为0，1，2三类的数据点分别绘制
# plt.scatter(X[y_pred==0,0],X[y_pred==0,1],marker='*')
# plt.scatter(X[y_pred==1,0],X[y_pred==1,1],marker='^')
# plt.scatter(X[y_pred==2,0],X[y_pred==2,1],marker='o')
# plt.axis('equal')
# plt.show()

#DBSCAN算法
# X1,y1=datasets.make_circles(n_samples=2000,factor=.6,noise=.05,random_state=0)
# X2,y2=datasets.make_blobs(n_samples=2000,n_features=2,centers=[[1.1,1.1]],cluster_std=[[.1]],random_state=0)
# X=np.concatenate((X1,X2))
# y_pred=DBSCAN(eps=0.1,min_samples=10).fit_predict(X)
# #将类别划分为0，1，2三类的数据点分别进行绘制
# plt.scatter(X[y_pred==0,0],X[y_pred==0,1],marker='*')
# plt.scatter(X[y_pred==1,0],X[y_pred==1,1],marker='^')
# plt.scatter(X[y_pred==2,0],X[y_pred==2,1],marker='o')
# plt.axis('equal')
# plt.show()