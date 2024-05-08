# Author:司震
# Create: 2024/4/16
# Description: 第9天练习：K-近邻
import numpy as np
from sklearn.neighbors import KNeighborsClassifier #导入K近邻分类器
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#采用决策树进行手写字符识别，数据规范化、Z-score规范化、min-max规范化对比
# digit = datasets.load_digits()
# x=digit.data
# y=digit.target
# #z-score规范化
# std1=StandardScaler()
# x_std=std1.fit_transform(x)
# #min-max规范化
# std2=MinMaxScaler()
# x_min_max=std2.fit_transform(x)
# ntrain=1000 #训练样本数量
# rate,rate_std,rate_min_max=[],[],[]
# for i in range(6,15):
#     #未规范化
#     clf=DecisionTreeClassifier(max_depth=i).fit(x[:ntrain,:],y[:ntrain])
#     rate.append(clf.score(x[ntrain:,:],y[ntrain:]))
#     #z-score规范化
#     clf_std=DecisionTreeClassifier(max_depth=i).fit(x_std[:ntrain,:],y[:ntrain])
#     rate_std.append(clf_std.score(x_std[ntrain:,:], y[ntrain:]))
#     #min-max规范化
#     clf_min_max = DecisionTreeClassifier(max_depth=i).fit(x_min_max[:ntrain,:], y[:ntrain])
#     rate_min_max.append(clf_min_max.score(x_min_max[ntrain:,:], y[ntrain:]))
# plt.plot(range(6,15),rate,'-',range(6,15),rate_std,'--',range(6,15),rate_min_max,'-.')
# plt.xlabel('depth of decison tree')
# plt.ylabel('accuracy')
# plt.legend(['raw data','zscore','min_max'])
# plt.show()

#采用K近邻算法对鸢尾花进行分类
#载入鸢尾花数据集
# iris = datasets.load_iris()
# #将样本与标签分开
# x = iris['data']
# y = iris['target']
# #划分数据集
# np.random.seed(10)#为了避免随机样本分类产生不同的结果，设置随机数种子
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)#训练集和测试集8：2
# acc=[]
# for i in range(1,100,2):
#     clf = KNeighborsClassifier(n_neighbors=i,p=2,metric='minkowski')#使用欧氏距离
#     clf.fit(x_train,y_train)
#     acc.append(clf.score(x_test,y_test))
# plt.plot(range(1,100,2),acc,'-*')
# plt.xlabel('K')
# plt.ylabel('accuracy')
# plt.show()






