# Author:司震
# Create: 2024/4/26
# Description: 第3天练习：手写字符分类
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
from pathlib import Path
import requests
import pickle
import gzip
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

DATA_PATH=Path('F:/ML_practice/ML_Python/Source')
PATH=DATA_PATH/'mnist'
PATH.mkdir(parents=True,exist_ok=True)
#URL='http://deeplearning.net/data/mnist/'
FILENAME='mnist.pkl.gz'
#下载
# if not (PATH/FILENAME).exists():
#     content=requests.get(URL+FILENAME).content
#     (PATH/FILENAME).open('wb').write(content)
#解压
with gzip.open((PATH/FILENAME).as_posix(),'rb') as f:
    ((x_train,y_train),(x_valid,y_valid),_)=pickle.load(f,encoding='latin-1')
plt.imshow(x_train[0].reshape((28,28)),cmap='gray')

#print(x_train.shape)

x_train,y_train,x_valid,y_valid=map(
    torch.tensor,(x_train,y_train,x_valid,y_valid)
)
n,c=x_train.shape
x_train,x_train.shape,y_train.min(),y_train.max()
# print(x_train,y_train)
# print(x_train.shape)
# print(y_train.min(),y_train.max())

#nn.function的用法
loss_func=F.cross_entropy
def model(xb):
    return xb.mm(weights)+bias
bs=64
xb=x_train[0:bs]
yb=y_train[0:bs]
weights=torch.randn([784,10],dtype=torch.float,requires_grad=True)
bs=64
bias=torch.zeros(10,requires_grad=True)
#print(loss_func(model(xb),yb))

#nn.Module的用法
class Mnist_NN(nn.Module):
   def __init__(self):
       super().__init__()
       self.hidden1=nn.Linear(784,128)
       self.hidden2=nn.Linear(128,256)
       self.out  =  nn.Linear(256,10)

   def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
net=Mnist_NN()
print(net)
#打印我们定义好名字里的权重和偏置项
#for name,parameter in net.named_parameters():
    #print(name,parameter,parameter.size())

train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader(train_ds,batch_size=bs,shuffle=True)

valid_ds = TensorDataset(x_valid,y_valid)
valid_dl = DataLoader(valid_ds,batch_size=bs*2)
def get_data(train_ds,valid_ds,bs):
    return (
        DataLoader(train_ds,batch_size=bs,shuffle=True),
        DataLoader(valid_ds,batch_size=bs*2)
    )
def fit(steps,model,loss_func,opt,train_dl,valid_dl):
    for step in range(steps):
        model.train()
        for xb,yb in train_dl:
            loss_batch(model,loss_func,xb,yb,opt)
        model.eval()
        with torch.no_grad():
            losses,nums=zip(
                *[loss_batch(model,loss_func,xb,yb)for xb,yb in valid_dl]
            )
        val_loss=np.sum(np.multiply(losses,nums))/np.sum(nums)
        print('当前step'+str(step),'验证机损失：'+str(val_loss))

def get_model():
    model = Mnist_NN()
    return model,optim.SGD(model.parameters(),lr=0.001)

def loss_batch(model,loss_func,xb,yb,opt=None):
    loss=loss_func(model(xb),yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(),len(xb)

train_dl,valid_dl=get_data(train_ds,valid_ds,bs)
model,opt = get_model()
fit(25,model,loss_func,opt,train_dl,valid_dl)