# Author:SiZhen
# Create: 2024/6/4
# Description: pytorch实现shufflenetV1
import torch
import torch.nn as nn
import  torch.nn.functional as F
from  torch.nn import init
from collections import OrderedDict

# 1x1卷积（降维/升维）
def conv1x1(in_chans,out_chans,n_groups=1):
    return nn.Conv2d(in_chans,out_chans,kernel_size=1,stride=1,groups=n_groups)

#3x3深度卷积
def conv3x3(in_chans,out_chans,stride,n_groups=1):
    #不管步长为多少，填充总为1
    return nn.Conv2d(in_chans,out_chans,kernel_size=3,padding=1,stride=stride,groups=n_groups)

#通道混洗
def channel_shuffle(x,n_groups):
    #获得特征图所有维度的数据
    batch_size,chans,height,width=x.shape
    #对特征通道进行分组
    chans_group = chans//n_groups
    #reshape新增特征图的维度
    x = x.view(batch_size,n_groups,chans_group,height,width)
    #通道混洗（将输入张量的指定维度进行交换）
    x = torch.transpose(x,1,2).contiguous()
    #reshape降低特征图的维度
    x = x.view(batch_size,-1,height,width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self,in_chans,out_chans,stride,n_groups=1):
        super(ShuffleUnit, self).__init__()
        #1x1分组卷积降维后的维度
        self.bottle_chans = out_chans//4
        #分组卷积的分组数
        self.n_groups = n_groups
        #是否进行下采样
        if stride ==1:
            #不进行下采样，分支和主干特征形状完全一致，直接执行add相加
            self.end_op = 'Add'
            self.out_chans = out_chans
        elif stride==2:
            #进行下采样，分支和主干特征形状完全一致，分支也需进行下采样，而后再进行concat拼接
            self.end_op = 'Concat'
            self.out_chans = out_chans-in_chans

        #1x1 卷积进行降维
        self.unit_1 = nn.Sequential(conv1x1(in_chans,self.bottle_chans,n_groups=n_groups),
                                    nn.BatchNorm2d(self.bottle_chans),
                                    nn.ReLU())

        #3x3深度卷积进行特征提取
        self.unit_2 = nn.Sequential(conv3x3(self.bottle_chans,self.bottle_chans,stride,n_groups=n_groups),
                                    nn.BatchNorm2d(self.bottle_chans))

        #1x1 卷积进行升维
        self.unit_3 = nn.Sequential(conv1x1(self.bottle_chans,self.out_chans,n_groups=n_groups),
                                    nn.BatchNorm2d(self.out_chans))

        self.relu = nn.ReLU(inplace=True)

    def forward(self,inp):
        #分支的处理方式（是否需要下采样）
        if self.end_op == 'Add':
            residual = inp
        else:
            residual = F.avg_pool2d(inp,kernel_size=3,stride=2,padding=1)

        x = self.unit_1(inp)
        x = channel_shuffle(x,self.n_groups)
        x = self.unit_2(x)
        x = self.unit_3(x)
        #分支与主干融合的方式
        if self.end_op == 'Add':
            return self.relu(residual +x)
        else:
            return self.relu(torch.cat((residual,x),1))

class ShuffleNetV1(nn.Module):
    def __init__(self,n_groups,n_classes,stage_out_chans):
        super(ShuffleNetV1, self).__init__()
        #输入通道
        self.in_chans = 3
        #分组组数
        self.n_groups = n_groups
        #分类个数
        self.n_classes = n_classes

        self.conv1 = conv3x3(self.in_chans,24,2)
        self.maxpool = nn.MaxPool2d(3,2,1)

        #Stage 2
        op = OrderedDict()
        """
    op = OrderedDict() 这行代码创建了一个有序字典（Ordered Dictionary）对象。
    有序字典是Python中的一个数据结构，它类似于常规的字典，但保持了元素插入的顺序。
    这意味着当你遍历有序字典时，元素会按照你插入它们的顺序被访问，
    这对于某些需要保持操作或层顺序的应用场景非常有用，比如在定义神经网络模型时。
    在PyTorch的上下文中，OrderedDict 常常用于构建nn.Sequential模型时，来确保添加到模型中的层（模块）按照添加的顺序被应用。
         """
        unit_prefix = 'stage_2_unit_'
        #每个Stage的首个基础单元都需要进行下采样，其他单元不需要
        op[unit_prefix+'0'] = ShuffleUnit(24,stage_out_chans[0],2,self.n_groups)
        for i in range(3):
            op[unit_prefix+str(i+1)]=ShuffleUnit(stage_out_chans[0],stage_out_chans[0],1,self.n_groups)
        self.stage2 = nn.Sequential(op)

        #Stage 3
        op = OrderedDict()
        unit_prefix='stage_3_unit_'
        op[unit_prefix+'0'] = ShuffleUnit(stage_out_chans[0],stage_out_chans[1],2,self.n_groups)
        for i in range(7):
            op[unit_prefix+str(i+1)] = ShuffleUnit(stage_out_chans[1],stage_out_chans[1],1,self.n_groups)
        self.stage3 = nn.Sequential(op)

        #Stage 4
        op = OrderedDict()
        unit_prefix = 'stage_4_unit_'
        op[unit_prefix+'0'] = ShuffleUnit(stage_out_chans[1],stage_out_chans[2],2,self.n_groups)
        for i in range(3):
            op[unit_prefix+str(i+1)] = ShuffleUnit(stage_out_chans[2],stage_out_chans[2],1,self.n_groups)
        self.stage4 = nn.Sequential(op)

        #全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        #全连接层
        self.fc = nn.Linear(stage_out_chans[-1],self.n_classes)
        #权重初始化
        self.init_params()

    #权重初始化
    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

#不同分组数对应的通道数也不同
stage_out_chans_list = [[144,288,576],[200,400,800],[240,480,960],
                        [272,544,1088],[384,768,1536]]

def shufflenet_v1_groups1(n_groups=1,n_classes=1000):
    model = ShuffleNetV1(n_groups=n_groups,n_classes=n_classes,
                         stage_out_chans=stage_out_chans_list[n_groups-1])
    return model

def shufflenet_v1_groups2(n_groups=2,n_classes=1000):
    model = ShuffleNetV1(n_groups=n_groups,n_classes=n_classes,
                         stage_out_chans=stage_out_chans_list[n_groups-1])
    return model

def shufflenet_v1_groups3(n_groups=3,n_classes=1000):
    model = ShuffleNetV1(n_groups=n_groups,n_classes=n_classes,
                         stage_out_chans=stage_out_chans_list[n_groups-1])
    return model

def shufflenet_v1_groups4(n_groups=4,n_classes=1000):
    model = ShuffleNetV1(n_groups=n_groups,n_classes=n_classes,
                         stage_out_chans= stage_out_chans_list[n_groups-1])
    return model

def shufflenet_v1_groupsother(n_groups=8,n_classes=1000):
    #groups>4
    modelother = ShuffleNetV1(n_groups=n_groups,n_classes=n_classes,
                         stage_out_chans=stage_out_chans_list[-1])
    return modelother
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = shufflenet_v1_groups1().to(device)
    model2 = shufflenet_v1_groups2().to(device)
    model3 = shufflenet_v1_groups3().to(device)
    model4 = shufflenet_v1_groups4().to(device)
    modelother = shufflenet_v1_groupsother().to(device)



















