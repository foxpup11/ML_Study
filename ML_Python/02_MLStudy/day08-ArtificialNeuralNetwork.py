# Author:司震
# Create: 2024/4/16
# Description: 第8天练习：人工神经网络
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#基于人工神经网络的手写字符分类
# my_data = load_digits()
# x = my_data.data
# y = my_data.target#获取数据
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
# score_logistic=[]#sigmoid函数
# score_relu=[]#线性整流函数
# score_tanh=[]#双曲正切函数
# for i in range(10,50,2):
#     clf_logistic=MLPClassifier(hidden_layer_sizes=(i,),activation='logistic',random_state=0,max_iter=2000)
#     clf_relu=MLPClassifier(hidden_layer_sizes=(i,),activation='relu',random_state=0,max_iter=2000)
#     clf_tanh=MLPClassifier(hidden_layer_sizes=(i,),activation='tanh',random_state=0,max_iter=2000)
#     score_logistic.append(clf_logistic.fit(X_train,y_train).score(X_test,y_test))
#     score_relu.append(clf_relu.fit(X_train,y_train).score(X_test,y_test))
#     score_tanh.append(clf_tanh.fit(X_train,y_train).score(X_test,y_test))
# plt.plot(range(10,50,2),score_logistic,'r-.',range(10,50,2),score_relu,'r--*',range(10,50,2),score_tanh,'b-.o')
# plt.xlabel('numbers of hidden layers')
# plt.ylabel('accuracy')
# plt.legend(['logistic','relu','tanh'])
# plt.show()
