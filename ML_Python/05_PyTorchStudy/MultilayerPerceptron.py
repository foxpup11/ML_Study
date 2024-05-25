# Author:SiZhen
# Create: 2024/5/24
# Description: pytorch实现一个处理分类任务的两层感知机
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

#本文件没有进行数据预处理，本程序主要目的是用pytorch理解多层感知机的流程机制


#定义多层感知机模型
class MLP(nn.Module):
    #重载初始化函数
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim,hidden_dim) #第一层线性变换
        self.layer2 = nn.Linear(hidden_dim,hidden_dim) #第二层线性变换
        self.output_layer = nn.Linear(hidden_dim,output_dim) #输出层
        self.activation = nn.ReLU() #选择Relu作为激活函数
    #定义前向传播
    def forward(self,x):
        x = self.activation(self.layer1(x)) #第一层的前向传播
        x = self.activation(self.layer2(x)) #第二层的前向传播
        x = self.output_layer(x) #输出层前向传播
        return x

input_dim = 128 #输入维度
hidden_dim = 64  #隐藏层维度
# output_dim = len(torch.unique(y_train)) #类别数
output_dim = 2

#实例化多层感知机
model = MLP(input_dim,hidden_dim,output_dim)
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss() #对于分类任务，常用的损失函数是交叉熵损失
optimizer = optim.SGD(model.parameters(),lr=0.01) #使用随机梯度下降进行优化

#准备数据加载器
# dataset = TensorDataset(X_train,y_train)
# dataloader = DataLoader(dataset,batch_size=32,shuffle=True) #使用mini-batch训练

#训练模型
num_epochs =100 #迭代次数
# for epoch in range(num_epochs):#每次迭代
#     for inputs,targets in dataloader: #每个mini-batch进行训练
#         #前向传播
#         outputs=model(inputs)
#         #计算损失
#         loss = criterion(outputs,targets)
#         #反向传播和优化
#         optimizer.zero_grad() #清零梯度,防止梯度爆炸
#         loss.backward()   #反向传播计算梯度，pytorch自动进行
#         optimizer.step()  #更新权重
#
#     print(f"epoch[{epoch+1}/{num_epochs}],Loss:{loss.item():.4f}")

