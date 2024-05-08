# Author:司震
# Create: 2024/4/16
# Description: 第10天练习：集成学习
import numpy as np
import matplotlib
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier #导入决策树作为基分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
matplotlib.rcParams['font.family']='SimHei'#加中文字体

#利用AdaBoost分类器和GBDT分类器解决手写字符分类问题
# digit = datasets.load_digits() #读入手写字符数据集
# x = digit.data   #将数据集的特征放入X
# y = digit.target #将数据集的类别标签放入y
# ntrain = 1000 #训练样本的数量
# x_train =x[:ntrain,:] #训练样本集X
# x_test = x[ntrain:,]  #测试样本集X
# y_train = y[:ntrain]
# y_test = y[ntrain:]
# result = []#用于存放测试集分类的准确率
# for i in range(2,15):
#     #采用决策树作为基分类器，决策树的最大深度设置为变量i,取值范围是[2,14]
#     base_calssifer = DecisionTreeClassifier(criterion='gini',max_depth=i)
#     #ADAbosst集成学习,采用100个分类器
#     #clf=AdaBoostClassifier(base_estimator=base_calssifer,n_estimators=100)
#     #GBDT分类器
#     clf =GradientBoostingClassifier(max_depth=i,n_estimators=100)
#     clf.fit(x_train,y_train) #训练
#     result.append(clf.score(x_test,y_test))#训练好的模型在测试集上预测并评分
# plt.plot(range(2,15),result,'b-*')  #画图
# plt.xlabel('depth of decision tree')
# plt.ylabel('accuracy')
# plt.show()


#基于GBDT的波士顿房价预测
# house_data = datasets.load_boston()
# x =house_data.data
# y =house_data.target
# n_train = 400#训练样本的数量
# #数据集分割
# x_train=x[:n_train]
# x_test=x[n_train:]
# y_train=y[:n_train]
# y_test=y[n_train:]
# pos=np.arange(1,11)#画柱状图的起始位置
# model=GradientBoostingRegressor().fit(x_train,y_train)#建模、训练
# y_predict = model.predict(x_test)#预测房价
# plt.bar(pos,y_test[:10],width=0.3,hatch='**')#hatch为填充图案
# plt.bar(pos+0.3,y_predict[:10],width=0.3)
# plt.xlabel('测试样本编号')
# plt.ylabel('价格/万美元')
# plt.legend(['房价真实值','房价预测值'])
# plt.show()

#bagging和随机森林手写字符识别效果对比
# digit = datasets.load_digits()
# x=digit.data
# y=digit.target
# ntrain=1000
# x_train=x[:ntrain]
# x_test=x[ntrain:]
# y_train =y[:ntrain]
# y_test =y[ntrain:]
# result_bagging=[]#用于存放测试集分类的准确率
# result_rf=[]#用于存放随机森林的结果
# for i in range(2,15):
#     #bagging方法
#     clf_bagging=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=i),n_estimators=100)
#     clf_bagging.fit(x_train,y_train)
#     result_bagging.append(clf_bagging.score(x_test,y_test))
#     #随机森林方法
#     clf_rf=RandomForestClassifier(max_depth=i,n_estimators=100)
#     clf_rf.fit(x_train,y_train)#训练
#     result_rf.append(clf_rf.score(x_test,y_test))
# plt.plot(range(2,15),result_bagging,'b-',range(2,15),result_rf,'b-.')
# plt.xlabel('决策树深度')
# plt.ylabel('准确率')
# plt.legend(['bagging','随机森林'])
# plt.show()