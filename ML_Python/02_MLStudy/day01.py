#2024/04/09 数据集划分和模型评价方法
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import  load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve

#分层留出法
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size,random_state,stratify,shuffle)

#交叉验证法
#kf = KFold(n_splits,shuffle,random_state)

#K-折交叉验证数据（K-Fold），以鸢尾花数据集为例
# iris = load_iris()#加载鸢尾花数据
# X = iris['data']
# y = iris['target']
# kf = KFold(n_splits = 5,shuffle = False,random_state = None)#5折
# for train_index,test_index in kf.split(X):
#     print("训练样本索引：",train_index,'测试样本索引：',test_index)#产生每次的测试集和训练集
#     X_train,X_test = X[train_index],X[test_index]
#     y_train,y_test = y[train_index],y[test_index]
#产生了五个模型

#自助法

#混淆矩阵
# y_true = [2,0,2,2,0,1]#真实值
# y_pred = [0,0,2,2,0,2]#预测值
# matrix = confusion_matrix(y_true,y_pred)#使用混淆矩阵
# print(matrix)
# precision_score = precision_score(y_true,y_pred,average='weighted')#计算查准率
# recall_score = recall_score(y_true,y_pred,average='weighted')#计算查全率
# f1 = f1_score(y_true,y_pred,average='weighted')#计算F1指标
# print('查准率是{:.2f},查全率是{:.2f},F1指标是{}'.format(precision_score,recall_score,f1))

#平均绝对误差(MAE)
# y = np.array([38,38.3,38.7,35.5,38.7])
# y_predict = np.array([36.2,38.9,39.0,29.9,39.2])
# mae = np.fabs(y-y_predict).mean()
# print(mae)

#均方误差（MSE）
# y = np.array([38,38.3,38.7,35.5,38.7])
# y_predict = np.array([36.2,38.9,39.0,29.9,39.2])
# mse = np.mean((y-y_predict)**2)
# print(np.round(mse,2))

#均方根误差（RMSE）
# y = np.array([38,38.3,38.7,35.5,38.7])
# y_predict = np.array([36.2,38.9,39.0,29.9,39.2])
# rmse = np.sqrt(np.mean((y-y_predict)**2))
# print(np.round(rmse,2))

#总平方和(SST)和回归平方和（SSR）计算R²
# mean_absolute_error(y_test,y_predict)
# mean_squared_error(y_test,y_predict)
# r2_score(y_test,y_predict)


#2x2 混淆矩阵
# y_true = [1,0,1,1,0,1]#真实值
# y_pred = [0,0,1,1,0,1]#预测值
# matrix = confusion_matrix(y_true,y_pred)#使用混淆矩阵
# print(matrix)

#ROC曲线


