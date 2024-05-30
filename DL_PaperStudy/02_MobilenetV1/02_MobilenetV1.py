# Author:SiZhen
# Create: 2024/5/28
# Description:MobileNetV1网络模型
import torch.nn as nn
import torch

class MobileNetV1(nn.Module):
    def __init__(self,ch_in,n_classes):
        super(MobileNetV1,self).__init__()

        #定义普通卷积，BN,激活模块
        def conv_bn(inp,oup,stride):
            return nn.Sequential(
                nn.Conv2d(inp,oup,3,stride,1,bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        #定义DW,PW卷积模块
        def conv_dw(inp,oup,stride):
            return nn.Sequential(
                #dw.关键是groups=inp时，就形成了深度卷积，即每个输入通道都有一个单独的卷积核（深度卷积核）与其匹配，且不共享卷积核。
                nn.Conv2d(inp,inp,3,stride,1,groups=inp,bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                #pw
                nn.Conv2d(inp,oup,1,1,0,bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(ch_in,32,2),
            conv_dw(32,64,1),
            conv_dw(64,128,2),
            conv_dw(128,128,1),
            conv_dw(128,256,2),
            conv_dw(256,256,1),
            conv_dw(256,512,2),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,1024,2),
            conv_dw(1024,1024,1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024,n_classes)

    #定义前向传播
    def forward(self,x):
        x = self.model(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return x
if __name__=="__main__":
    #model check
    model = MobileNetV1(ch_in=3,n_classes=5)
    print(model)
    random_data = torch.rand([1,3,224,224])
    result = model(random_data)
    print(result)


