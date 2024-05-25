# Author:司震
# Create: 2024/4/24
# Description: 第1天练习:torch基础练习,自动求导，反向传播，线性回归
import torch
import torch.nn as nn
import numpy as np
#print(torch.__version__) #查询pytorch的版本（CPU版）
# x=torch.empty(3,5)
# print(x)

#线性回归
x_values=[i for i in range(11)]
x_train=np.array(x_values,dtype=np.float32)
x_train=x_train.reshape(-1,1) #-1是指根据列数自动计算行数
#print(x_train.shape)

y_values=[2*i +1 for i in x_values]
y_train=np.array(y_values,dtype=np.float32)
y_train=y_train.reshape(-1,1)
#print(y_train.shape)

#线性回归模型.其实线性回归就是一个不加激活函数的全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        out=self.linear(x)
        return out
input_dim=1
output_dim=1

model=LinearRegressionModel(input_dim,output_dim)
#print(model)

#使用GPU去训练
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)

#指定好参数和损失函数
epochs=1000
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate) #优化器：SGD：随机梯度下降（stochastic gradient descent）
criterion=nn.MSELoss() #MSE：均方误差损失函数

#训练模型
for epoch in range(epochs):
    epoch +=1
    #注意转成tensor类型(np的数组是array类型的)
    #inputs=torch.from_numpy(x_train)
    #labels=torch.from_numpy(y_train)
    #若使用GPU去训练：
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    #梯度要清零每一次迭代（如果不清零每次梯度计算会累加起来）
    optimizer.zero_grad()
    #前向传播
    outputs=model(inputs)
    #计算损失
    loss=criterion(outputs,labels)
    #反向传播
    loss.backward()
    #更新权重参数
    optimizer.step()
    #if epoch %50==0:
        #print('epoch{},loss{}'.format(epoch,loss.item()))
#测试模型预测结果
predicted=model(torch.from_numpy(x_train).requires_grad_()).data.numpy
#print(predicted)
#模型的保存与提取
torch.save(model.state_dict(),'model.pkl')
print(model.load_state_dict(torch.load('model.pkl')))

