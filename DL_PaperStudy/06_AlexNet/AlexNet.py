# Author:SiZhen
# Create: 2024/6/2
# Description: pytorch实现Alexnet模型
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(128,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048), #全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_classes),
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x