# Author:Sizhen
# Create: 2024/4/24
# Description: pytoch气温预测实战（回归）
import sklearn
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch import tensor
import torch.optim as optim
import torch.nn.functional as F
from torchvision import  datasets,transforms
import matplotlib.pyplot as plt


#hub模块(了解)
# model = torch.hub.load("pytorch/vision:v0.4.2",'deeplabv3_resnet101',pretrained=True)
# model.eval()

#拿数据
features=pd.read_csv(r"F:/ML_Python/Source/temps.csv")
#查看数据长什么样子
# print(features.head())#查看前几行
# print('数据维度',features.shape)

#处理时间数据,分别得到年，月，日
import datetime
years=features['year']
months=features['month']
days=features['day']
#datetime格式
dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
dates=[datetime.datetime.strptime(date,'%Y-%m-%d')for date in dates]
# print(dates[:5])#展示看看

#准备画图，指定默认风格
plt.style.use('fivethirtyeight')
#设置布局
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
fig.autofmt_xdate(rotation=45)
#标签值(实际值)
ax1.plot(dates,features['actual'])
ax1.set_xlabel('');ax1.set_ylabel('Temperature');ax1.set_title('MaxTemp')
#昨天气温
ax2.plot(dates,features['temp_1'])
ax2.set_xlabel('');ax2.set_ylabel('Temperature');ax2.set_title('Previous Max Temp')
#前天气温
ax3.plot(dates,features['temp_2'])
ax3.set_xlabel('');ax3.set_ylabel('Temperature');ax3.set_title('Two Days Prior Max Temp')
#朋友猜的气温
ax4.plot(dates,features['friend'])
ax4.set_xlabel('');ax4.set_ylabel('Temperature');ax4.set_title('Friend Estimate')

plt.tight_layout(pad =2)
#plt.show()

#将week中的字符串转为编码(独热编码)
features=pd.get_dummies(features)
#print(features.head(5))

#将X和Y值分离
labels= np.array(features['actual']) #定义标签
features=features.drop('actual',axis=1)#在特征中去掉标签
feature_list=list(features.columns)#名字单独保存，以备后患
features=np.array(features)#转换成合适的格式
#print(features.shape)

#数据标准化（数据有大有小，需要归一化）
input_features=preprocessing.StandardScaler().fit_transform(features)
#print(input_features[0])\

#用torch搭建神经网络模型
x=torch.tensor(input_features,dtype=float)
y=torch.tensor(labels,dtype=float)
#权重参数初始化
weights=torch.randn((14,128),dtype=float,requires_grad=True) #第一个隐层，128个神经元
biases=torch.randn(128,dtype=float,requires_grad=True) #偏差
weights2=torch.randn((128,1),dtype=float,requires_grad=True) #第二个隐层，得到一个值
biases2=torch.randn(1,dtype=float,requires_grad=True) #偏差

learning_rate=0.001
losses=[]
for i in range(1000):
    hidden=x.mm(weights)+biases #计算隐层，mm即matrix multiply
    hidden=torch.relu(hidden) #加入激活函数
    predictions=hidden.mm(weights2)+biases2 #预测结果
    loss=torch.mean((predictions-y)**2) #计算损失
    losses.append(loss.data.numpy())
    #打印损失值
    #if i %100==0:
        #print('loss:',loss)
    #反向传播计算
    loss.backward()
    #更新参数
    weights.data.add_(-learning_rate*weights.grad.data)
    biases.data.add_(-learning_rate*biases.grad.data)
    weights2.data.add_(-learning_rate*weights2.grad.data)
    biases2.data.add_(-learning_rate*biases2.grad.data)
    #每次迭代记得清空
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

#构建更加简单的神经网络模型
input_size=input_features.shape[1]
hidden_size=127
out_size=1
batch_size=16
my_nn=torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,out_size)
)
cost=torch.nn.MSELoss(reduction='mean')
optimizer=torch.optim.Adam(my_nn.parameters(),lr=0.001)
#训练网络
losses=[]
for i in range(1000):
    batch_loss=[]
    #MINI-Batch方法来进行训练
    for start in range(0,len(input_features),batch_size):
        end=start+batch_size if start + batch_size<len(input_features) else len (input_features)
        xx = torch.tensor(input_features[start:end],dtype=torch.float,requires_grad=True)
        yy = torch.tensor(labels[start:end],dtype=torch.float,requires_grad=True)
        prediction=my_nn(xx)
        loss = cost(prediction,yy)#计算误差
        optimizer.zero_grad()#梯度清零操作
        loss.backward(retain_graph=True)
        optimizer.step()#参数更新
        batch_loss.append(loss.data.numpy())
        #打印损失
   # if i % 100 ==0:
      #  losses.append(np.mean(batch_loss))
       # print(i,np.mean(batch_loss))
#预测训练结果
x=torch.tensor(input_features,dtype=torch.float)
predict=my_nn(x).data.numpy()
#转换日期格式
dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
dates=[datetime.datetime.strptime(date,'%Y-%m-%d')for date in dates]
#创建一个表格来存日期和其对应的标签数值
true_data=pd.DataFrame(data={'date':dates,'actual':labels})
#同理，再创建一个来存日期和其对应的模型预测值
months=features[:,feature_list.index('month')]
days=features[:,feature_list.index('day')]
years=features[:,feature_list.index('year')]

test_dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day))for year ,month ,day in zip(years,months,days)]
test_dates=[datetime.datetime.strptime(date,'%Y-%m-%d')for date in test_dates]
predictions_data=pd.DataFrame(data={'date':test_dates,'prediction':predict.reshape(-1)})
#真实值
plt.plot(true_data['date'],true_data['actual'],'b-',label='actual')
#预测值
plt.plot(predictions_data['date'],predictions_data['prediction'],'ro',label='prediction')
plt.xticks(rotation=60);
plt.legend()
plt.xlabel('Date');plt.ylabel('Maximum Temperature(F)');plt.title('Actual and Predicted Values');
plt.show()





