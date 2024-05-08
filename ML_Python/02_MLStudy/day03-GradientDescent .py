# Author:司震
# Create: 2024/4/10
# Description: 第3天练习：梯度下降
# import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

# #数据集10个数据点
# m = 10
# #X矩阵
# X0=np.ones((m,1))
# X1 = np.array([4,8,9,8,7,12,6,10,6,9]).reshape(-1,1)
# X = np.hstack((X0,X1))
# #Y矩阵
# Y = np.array([9,20,22,15,17,23,18,25,10,20]).reshape(-1,1)
# #学习率
# alpha = 0.001
# #定义代价函数J（）
# def cost_function(theta,X,Y):
#     diff = np.dot(X,theta)-Y
#     return (1/2*m)*np.dot(diff.T,diff)
# #定义梯度函数
# def gradient_function(theta,X,Y):
#     diff = np.dot(X,theta)-Y
#     return np.dot(X.T,diff)/m
# #梯度下降法迭代
# def gradient_desent(X,Y,alpha):
#     theta = np.random.random(size=(2,1))
#     gradient = gradient_function(theta,X,Y)
#     while not all(abs(gradient)<= 10e-5):
#         theta =theta-alpha*gradient
#         gradient = gradient_function(theta,X,Y)
#     return theta
# #运行梯度下降
# theta = gradient_desent(X,Y,alpha)
# print('optimal:',theta)
# plt.figure(dpi=128,figsize=(10,6))
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# #样本散点图
# plt.scatter(X[:,1],Y.flatten(),s=100)
# plt.xlim(0,15)
# plt.ylim(0,35)
# #theta = (b;w)列向量
# #绘制回归曲线
# x_axis = np.linspace(0,15,200).reshape(-1,1)
# x = np.hstack((np.ones((200,1)),x_axis))
# y = np.dot(x,theta)
# plt.plot(x[:,1],y,linestyle="-",linewidth = 3,color = 'red')
# plt.text(10,18,r'$y={}x+{}$'.format(round(theta[1,0],3),round(theta[0,0],3)),fontsize = 17)
# plt.xlabel('广告费/万元',fontsize = 16)
# plt.ylabel('销售额/万元',fontsize = 16)
# plt.legend(["梯度下降法"],fontsize = 16)
# plt.show()

#sklearn自带的最小二乘线性回归和梯度下降预测销售额
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# X = np.array([4,8,9,8,7,12,6,10,6,9]).reshape(-1,1)
# y = np.array([9,20,22,15,17,23,18,25,10,20])
# #最小二乘法线性回归模型
# LS_reg = LinearRegression().fit(X,y)
# w1 = LS_reg.coef_
# b1 = LS_reg.intercept_
# #梯度下降法线性回归模型
# SGD_reg = SGDRegressor().fit(X,y)
# w2 = SGD_reg.coef_
# b2 = SGD_reg.intercept_
# #采用R2评价回归模型
# y_pred_ls = LS_reg.predict(X)
# score_r2_LS = r2_score(y,y_pred_ls)
# y_pred_SGD = SGD_reg.predict(X)
# score_r2_SGD = r2_score(y,y_pred_SGD)
# print("Least Square r2 score :{}\nSGD r2 scroe:{}".format(score_r2_LS,score_r2_SGD))
# #绘图
# fig = plt.figure(dpi=128,figsize=(10,6))
# plt.scatter(X,y,s=100)
# x_pred = np.linspace(0,15,200)
# y_ls = w1*x_pred+b1
# y_SGD = w2*x_pred +b2
# plt.plot(x_pred,y_ls,linestyle='-',linewidth = 4)
# plt.plot(x_pred,y_SGD,linestyle=':',linewidth = 4)
# plt.show()



