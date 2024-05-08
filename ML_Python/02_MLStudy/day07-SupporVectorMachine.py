# Author:司震
# Create: 2024/4/15
# Description: 第7天练习：支持向量机
from sklearn import datasets
from sklearn import svm
import  numpy as np
from sklearn.model_selection import train_test_split as split
import matplotlib.pyplot as plt


#采用svm.SVC 对鸢尾花数据进行分类
#导入数据
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# np.random.seed(30)
# X_train,X_test,y_train,y_test = split(X,y,test_size=0.3)#训练样本和测试样本比例7：3
# rbf_score,linear_score,poly_score =[],[],[]
#
# c_values=np.linspace(0.1,1,20)#惩罚系数的取值范围
# for c in c_values:
#     #高斯rbf核
#     clf_rbf = svm.SVC(C=c,kernel='rbf',random_state=100)
#     clf_rbf.fit(X_train,y_train)
#     rbf_score.append(clf_rbf.score(X_test,y_test))
#     #线性核函数
#     clf_linear = svm.SVC(C=c,random_state=100,kernel='linear')
#     clf_linear.fit(X_train,y_train)
#     linear_score.append(clf_linear.score(X_test,y_test))
#     #多项式核函数
#     clf_poly=svm.SVC(C=c,random_state=100,kernel='poly')
#     clf_poly.fit(X_train,y_train)
#     poly_score.append(clf_poly.score(X_test,y_test))
# #画图
# plt.plot(c_values,rbf_score,'--',c_values,linear_score,'-',c_values,poly_score,'-+')
# plt.legend(['rbf','linear','poly'])
# plt.xlabel('C')
# plt.ylabel('accuracy')
# plt.title('Iris classification by SVC')
# plt.show()