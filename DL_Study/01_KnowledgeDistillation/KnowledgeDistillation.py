# Author:SiZhen
# Create: 2024/5/25
# Description: pytorch实现知识蒸馏,使用MNIST数据集
import torch
from torch import nn
import  torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
# print(torch.cuda.is_available())
#设置随机数种子
torch.manual_seed(0)

#使用cuDNN加速卷积运算
# torch.backends.cudnn.benchmark = True
device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
#载入训练集
train_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
#载入测试集
test_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
#生成dataloader
train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader= DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)

#教师模型
class TeacherModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200,num_classes)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x = x.view(-1,784) # 自动计算有多少个样本
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

#训练教师模型
model = TeacherModel()
model = model.to(device)

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

#开始训练
epochs = 6
for epoch in range(epochs):
    model.train()
    #训练集上训练模型的权重
    for data ,targets in tqdm(train_loader): #tqdm用于生成进度条
         data = data.to(device)
         targets = targets.to(device)
         #前向传播
         preds = model(data)
         loss = criterion(preds,targets)

         #反向传播
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
    #测试集上评估模型性能
    model.eval()
    num_correct = 0
    num_samples = 0
    # 在评估模型时，我们通常不需要计算梯度（因为不进行反向传播和权重更新），这样做可以节省内存并加速计算过程
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices #找到预测概率最大的类别索引。.max(1) 表示按行查找最大值，.indices 则返回这些最大值对应的索引，即预测的类别标签。
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()#将张量转化为python标量
    model.train()
    print('epoch:{}\t accuracy:{:.4f}'.format(epoch+1,acc))

class StudentModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(StudentModel,self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,num_classes)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

teacher_model = model

#训练学生模型
model = StudentModel()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs = 3
for epoch in range(epochs):
    model.train()

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        #前向传播
        preds = model(data)
        loss = criterion(preds,targets)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #更新参数

    #测试集上评估模型的性能
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0) #predictions 张量的第一个维度的长度
        acc = (num_correct/num_samples).item()

    model.train()
    print('eopch:{}\t accuracy:{:.4f}'.format(epoch+1,acc))

student_model_scratch = model

#知识蒸馏训练学生模型

#准备预训练好的教师模型
teacher_model.eval()
#准备新的学生模型
model = StudentModel()
model = model.to(device)
model.train()

#蒸馏温度
Temperature = 7

hard_loss = nn.CrossEntropyLoss()
alpha = 0.3 #hard_loss的权重

soft_loss = nn.KLDivLoss(reduction="batchmean") #KL散度
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

#开始训练
epochs = 3
for epoch in range(epochs):
    #训练集上训练模型的权重
    for data,targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        #教师模型预测
        with torch.no_grad():
            teacher_preds = teacher_model(data)

        #学生模型预测
        student_preds = model(data)

        #计算hard_loss
        student_loss = hard_loss(student_preds,targets)

        #计算蒸馏后的预测结果及soft_loss
        distillation_loss = soft_loss(
            F.softmax(student_preds/Temperature,dim=1),
            F.softmax(teacher_preds/Temperature,dim=1)
        )

        #将hard_loss 和soft_loss 加权求和
        loss = alpha * student_loss + (1-alpha) * distillation_loss

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #更新权重

    #测试集上评估模型的性能
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds =model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    model.train()
    print('eopch:{}\t accuracy:{:.4f}'.format(epoch+1,acc))












