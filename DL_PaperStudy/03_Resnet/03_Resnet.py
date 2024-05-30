# Author:SiZhen
# Create: 2024/5/30
# Description: pytorch实现Resnet18和34
import torch
from torch import nn
from torch.nn import functional as F

#定义BasicBlock
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,
                 use_1x1conv=False,strides=1):
        super().__init__()
        #定义两层普通卷积
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3
                             ,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(input_channels,num_channels,kernel_size=3,
                               padding=1)
        #1x1卷积层进行下采样，从而升维
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,
                                kernel_size=1,stride=strides)
        else:
            self.conv3 = None
            self.bn1=nn.BatchNorm2d(num_channels)
            self.bn2=nn.BatchNorm2d(num_channels)
        #定义前向传播
        def forward(self,X):
            #卷积，批量归一化，激活函数拟合
            Y=F.relu(self.bn1(self.conv1(X)))
            #卷积，批量归一化，不激活
            Y =self.bn1(self.conv1(Y))
            #判定是否进行恒等映射
            if self.conv3:
                X=self.conv3(X)#恒等映射，控制通道数
            Y+=X #残差块加恒等映射

            return F.relu(Y) #线性激活，防止梯度消失

b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                   nn.BatchNorm2d(64),nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

#构建Resnet18
def resnet18(num_classes,in_channels=1):
    #定义残差块
    def resnet_block(in_channels,out_channels,num_residuals,
                     first_block=False):
        blk=[]
        for i in range(num_residuals):
            # 是否需要1x1卷积进行升维
            if i==0 and not first_block:
                blk.append(Residual(in_channels,out_channels,
                                    use_1x1conv=True,strides=2))
            else:
                blk.append(Residual(out_channels,out_channels))

        return nn.Sequential(*blk)

    net = nn.Sequential(b1)

    net.add_module("resnet_block1",resnet_block(64,64,2,first_block=True))
    net.add_module("resnet_block2",resnet_block(64,128,2))
    net.add_module("resnet_block3",resnet_block(128,256,2))
    net.add_module("resnet_block4",resnet_block(256,512,2))
    net.add_module("global_avg_pool",nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc",nn.Sequential(nn.Flatten(),
                                      nn.Linear(512,num_classes)))
    return net


#构建Resnet34
def resnet34(num_classes,in_channels=1):
    #定义残差块
    def resnet_block(in_channels,out_channels,num_residuals,
        first_block=False):

        blk = []
        for i in range(num_residuals):
            if i ==0 and not first_block:
                blk.append(Residual(in_channels,out_channels,
                                      use_1x1conv=True,strides=2))
            else:
                blk.append(Residual(out_channels,out_channels))
        return nn.Sequential(*blk)

        net = nn.Sequential(b1)

        net.add_module("resnet_block1",resnet_block(64,64,3,first_block=True))
        net.add_module("resnet_block2",resnet_block(64,128,4))
        net.add_module("resnet_block3",resnet_block(128,256,6))
        net.add_module("resnet_block4",resnet_block(256,512,3))
        net.add_module("global_avg_pool",nn.AdaptiveAvgPool2d((1,1)))
        net.add_module("fc",nn.Sequential(nn.Flatten(),
                                          nn.Linear(512,num_classes)))
        return net









