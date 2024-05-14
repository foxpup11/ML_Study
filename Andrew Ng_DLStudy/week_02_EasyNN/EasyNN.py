# Author:SiZhen
# Create: 2024/5/14
# Description: 构建具有单隐藏层的2类分类神经网络。
# 使用具有非线性激活功能激活函数，例如tanh。
# 计算交叉熵损失（损失函数）。
# 实现向前和向后传播。
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)
X,Y=load_planar_dataset()#加载数据集
#plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)#绘制散点图可视化数据集
# plt.show()
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1] #训练集的数量
# print(shape_X)
# print(shape_Y)
# print(m)

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T)
# plot_decision_boundary(lambda x:clf.predict(x),X,Y)#绘制决策边界
# plt.title("Logistic Regression")
# LR_predictions = clf.predict(X.T) #预测结果
# print('accuracy: %d'%float((np.dot(Y,LR_predictions)+np.dot(1-Y,1-LR_predictions))/float(Y.size)*100)+"%")
# plt.show()

#定义神经网络结构
def layer_sizes(X,Y):
    n_x = X.shape[0] #输入层的特征个数
    n_h = 4 #隐藏层的神经单元个数
    n_y = Y.shape[0] #输出层

    return (n_x,n_h,n_y)

#初始化模型的参数
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))

    assert (W1.shape==(n_h,n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y,1))

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters


#前向传播
def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    #前向传播计算A2
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    assert(A2.shape==(1,X.shape[1]))
    cache = {
        'Z1':Z1,
        'A1':A1,
        "Z2":Z2,
        "A2":A2 }
    return (A2,cache)

#计算损失
# logprobs = np.multiply(np.log(A2),Y)
# cost = -np.sum(logprobs)
def compute_cost(A2,Y,parameters):
    m=Y.shape[1] #样本个数
    W1=parameters["W1"]
    W2=parameters["W2"]

    #计算成本
    logprobs=logprobs=np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(logprobs)/m
    cost = float(np.squeeze(cost)) #squeeze方法的主要作用是移除数组中所有长度为1的维度

    assert(isinstance(cost,float))
    return cost

#反向传播
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2-Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2)) #注意multiply和dot的区别
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
    grads={
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }
    return grads

#参数更新
def update_parameters(parameters,grads,learing_rate=1.2):
    W1,W2=parameters["W1"],parameters["W2"]
    b1,b2=parameters["b1"],parameters["b2"]
    dW1,dW2=grads["dW1"],grads["dW2"]
    db1,db2=grads["db1"],grads["db2"]

    W1=W1-learing_rate*dW1
    W2=W2-learing_rate*dW2
    b1=b1-learing_rate*db1
    b2=b2-learing_rate*db2

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters

#把上面的流程整合进模型里面,要以正确的顺序使用先前的功能
def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    #训练模型，指定迭代次数
    for i in range(num_iterations):
        A2 , cache = forward_propagation(X,parameters)#前向传播
        cost = compute_cost(A2,Y,parameters) #计算损失函数
        grads = backward_propagation(parameters,cache,X,Y)#反向传播
        parameters = update_parameters(parameters,grads,learing_rate=0.5) #参数更新，指定学习率

        if print_cost:
            if i%1000==0:
                print("第",i,"次循环，成本为："+str(cost))
    return parameters

#预测结果
def predict(parameters,X):
    A2,cache =forward_propagation(X,parameters)
    predictions = np.round(A2)

    return predictions

#所有模块搭建结束，正式运行
parameters = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)
#绘制边界
# plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
# plt.title("Decision Boundary for hidden layer size "+str(4))
# predictions = predict(parameters,X)
# print('accuracy: %d'%float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)+"%")
# plt.show()


#更改隐藏层节点数量
plt.figure(figsize=(16,32))
hidden_layer_sizes=[1,2,3,4,5,20,50]#隐藏层数量
for i,n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5,2,i+1)
    plt.title('hidden layer of size%d'%n_h)
    parameters =nn_model(X,Y,n_h,num_iterations=5000)
    plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print("隐藏层节点数量：{},准确率：{}%".format(n_h,accuracy))
plt.show()


