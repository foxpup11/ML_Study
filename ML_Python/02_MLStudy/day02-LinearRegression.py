#第四章线性模型 2024/04/09
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
matplotlib.rcParams['font.family']='SimHei'


#一元线性回归例子
# x = np.array([4,8,9,8,7,12,6,10,6,9])
# y = np.array([9,20,22,15,17,23,18,25,10,20])
# plt.scatter(x,y,marker='o')
# plt.xlabel('广告费')
# plt.ylabel('销售额')
# plt.title("某公司广告费和销售额之间的散点图")
# plt.show()

#一元线性回归预测销售额
#加载数据
# x = np.array([4,8,9,8,7,12,6,10,6,9]).reshape(-1,1)
# y = np.array([9,20,22,15,17,23,18,25,10,20])
#线性模型训练
# reg = LinearRegression().fit(x,y)
#获取线性模型属性w,b
# w = reg.coef_
# b = reg.intercept_
# fig = plt.figure(dpi=128,figsize=(10,6))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#绘制训练样本散点图
# plt.scatter(x,y,s=100)
#绘制回归直线
#X轴数据
# x_pred = np.linspace(0,15,200)
#模型预测的数据
# y_pred = w*x_pred +b
# plt.plot(x_pred,y_pred,linestyle="--",linewidth=4)
# plt.xlim(0,15)
# plt.ylim(0,35)
# plt.text(10,18,r"$y={}x+{}$".format(round(w[0],3),round(b,3)),fontsize=16)
# plt.xlabel('广告费',fontsize=16)
# plt.ylabel('销售额',fontsize=16)
# plt.show()

#多元线性回归预测销售额
# data = pd.read_excel(r"path")
# x = data.iloc[:,0:3]#前三列作为X
# y = data.iloc[:,-1]#最后一列作为Y
# reg = LinearRegression().fit(x,y) #建立线性回归模型，并训练
# w = reg.coef_
# b = reg.intercept_
# print("估计系数：w1,w2,w3:",w)#输出估计出来的权重参数
# print("偏置项w0:",b)#输出偏置
# #预测
# y_pred = reg.predict(x)
# #采用R²评价回归模型
# score_r2=r2_score(y,y_pred)
# print("r2 score:",score_r2)
# w = np.around(w,decimals=3)
# print("多元线性回归的方程为：")
# print("y = {} +{}x1 +{}x2 + {}x3 ".format(round(b,2),w[0],w[1],w[2]))


