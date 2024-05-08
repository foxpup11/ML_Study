# # Author:司震
# # Create: 2024/4/27
# # Description: 第5天练习：pytorch项目实战:花朵识别
# import torchvision
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch import nn
# import torch.optim as optim
# from torchvision import transforms,models,datasets
# import  imageio
# import  time
# import warnings
# import random
# import sys
# import copy
# import json
# from PIL import  Image
#
# #数据读取
# data_dir='F:/ML_Python/Source/flower_data'
# train_dir = data_dir+'/train'
# valid_dir = data_dir+'/valid'
# #数据预处理。data_transforms中指定了所有图像预处理操作。图像增强。
# #imageFolder假设所有的文件按文件夹保存好，每个文件夹下存储同一类别的图片，文件夹的名字为分类的名字
# data_transforms={
#     'train':transforms.Compose([transforms.RandomRotation(45),  #随机旋转，-45到45度之间随便选
#          transforms.CenterCrop(224), #从中心开始裁剪
#          transforms.RandomHorizontalFlip(p=0.5) , #随机水平翻转 选择一个概率
#          transforms.RandomVerticalFlip(p=0.5),  #随机垂直翻转，选择一个概率
#          transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1), #参数依次为亮度，对比度，饱和度，色相
#          transforms.RandomGrayscale(p=0.225), #将概率转换为灰度率，三通道就是R=G=B
#          transforms.ToTensor(),
#          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  #均值，标准差。迁移学习，让学习效果更好
# ]),
#     'valid':transforms.Compose([transforms.Resize(256), #resize操作，统一规格
#          transforms.CenterCrop(224), #再裁剪成224x224格式的
#          transforms.ToTensor(),
#          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ]),
# }
# batch_size = 4
# #构建数据集
# image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])for x in ['train','valid']}
# dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True)for x in['train','valid']}
# dataset_sizes={x:len(image_datasets[x])for x in ['train','valid']}
# class_names=image_datasets['train'].classes
#
# #读取标签对应的实际名字
# with open('F:/ML_Python/Source/cat_to_name.json','r')as f:
#     cat_to_name =json.load(f)
# #print(cat_to_name)
#
# #展示下数据。注意tensor的数据需要转换成numpy的格式，而且需要还原回标准化的结果
# def im_convert(tensor):
#     #展示数据
#     image=tensor.to('cpu').clone().detach()
#     image=image.numpy().squeeze()
#     image=image.transpose(1,2,0)
#     image=image*np.array((0.229,0.224,0.225))+np.array((0.485,0.456,0.406))
#     image=image.clip(0,1)
#     return image
#
# fig =plt.figure(figsize=(20,12))
# columns=4
# rows = 2
#
# dataiter=iter(dataloaders['valid'])
# inputs,classes=dataiter.next()
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows,columns,idx+1,xticks=[],yticks=[])
#     ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
#     plt.imshow(im_convert(inputs[idx]))
# plt.show()
#
# #加载models中提供的模型，并且直接用训练好的权重当作初始化参数
# model_name = 'resnet'#可选的比较多['resnet','alexnet','vgg','squeezenet','densenet','inception']
#
# #是否用人家训练好的特征去做
# feature_extract=True
# train_on_gpu=torch.cuda.is_available() #是否用gpu去训练
# if not train_on_gpu:
#     print('CUDA is not available.Training on CPU...')
# else:
#     print('CUDA is available! Training on GPU')
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#
# def set_parameter_requires_grad(model,feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad=False
#
# model_ft=models.resnet152() #152层resnet网络
# #print(model_ft) #查看152层怎么做的
#
# #模型初始化
# def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
#     #选择合适的模型，不同模型的初始化方法稍微有点区别
#     model_ft=None
#     input_size = 0
#     if model_name=='resnet':
#         model_ft=models.resnet152(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft,feature_extract)
#         num_ftrs=model_ft.fc.in_features
#         model_ft.fc=nn.Sequential(nn.Linear(num_ftrs,102),
#                                   nn.LogSoftmax(dim=1))
#         input_size=224
#     else:
#         print('Invalid model name ,existing...')
#         exit()
#     return  model_ft,input_size
#
# model_ft,input_size=initialize_model(model_name,102,feature_extract,use_pretrained=True)
# model_ft=model_ft.to(device)#GPU计算
# filename='checkpoint.pth'#模型保存
# #是否训练所有层
# params_to_update=model_ft.parameters()
# print('Params to learn')
# if feature_extract:
#     params_to_update=[]
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad==True:
#             params_to_update.append(param)
#             print('\t',name)
# else:
#     for name,param in model_ft.named_paraeters():
#         if param.requires_grad==True:
#             print("\t",name)
# #优化器设置
# optimizer_ft = optim.Adam(params_to_update,lr=1e-2)
# scheduler = optim.lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1) #学习率每7个epoch衰减成原来的1/10
# #最后一层已经logsoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
# criterion=nn.NLLLoss()
#
# #训练模型
# def train_model(model,dataloaders,criterion,optimizer,num_epochs=25,is_inception=False,filename=filename):
#     since=time.time()
#     best_acc=0
#     model.to(device)
#     val_acc_history=[]
#     train_acc_history=[]
#     train_loss=[]
#     valid_losses=[]
#     LRs=[optimizer.param_groups[0]['lr']]
#     best_model_wts=copy.deepcopy(model.state_dict())
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch,num_epochs-1))
#         print('-'*10)
#         #训练和验证
#         for phase in ['trian','valid']:
#             if phase == 'train':
#                 model.train() #训练
#             else:
#                 model.eval() #验证
#             running_loss = 0.0
#             running_correct = 0
#             #把数据都读个遍
#             for inputs,labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 #清零
#                 optimizer.zero_grad()
#                 #只有训练的时候计算和更新梯度
#                 with torch.set_grad_enabled(phase=='train'):
#                     if is_inception and phase == 'train':
#                         outputs,aux_outputs=model(inputs)
#                         loss1=criterion(outputs,labels)
#                         loss2=criterion(aux_outputs,labels)
#                         loss = loss1+loss2*0.4
#                     else:#resnet执行的是这里
#                         outputs = model(inputs)
#                         loss=criterion(outputs,labels)
#                         _,preds=torch.max(outputs,1)
#                     #训练阶段更新权重
#                     if phase=='train':
#                         loss.backward()
#                         optimizer.step()
#                  #计算损失
#                  running_loss=loss.item()*inputs.size(0)
#                  running_correct += torch.sum(preds==labels.data)
#
#             epoch_loss =running_loss/len(dataloaders[phase].dataset)
#             epoch_acc = running_correct.double()/len(dataloaders[phase].dataset)
#             time_elapsed = time.time()-since
#             print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed)//60,time_elapsed%60))
#             print('{}Loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
#             #得到最好那次的模型
#             if phase =='valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts=copy.deepcopy(model.state_dict())
#                 state={
#                     "state_dict":model.state_dict(),
#                     'best_acc':best_acc,
#                     'optimizer':optimizer.state_dirt(),
#         }
#                 torch.save(state, filename)
#             if phase == 'valid':
#                 val_acc_history.append(epoch_acc)
#             valid_losses.append(epoch_loss)
#             scheduler.step(epoch_loss)
#             if phase == 'train':
#                 train_acc_history.append(epoch_acc)
#             train_losses.append(epoch_loss)
#
#             print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
#             LRs.append(optimizer.param_groups[0]['lr'])
#             print()
#
#             time_elapsed = time.time() - since
#             print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#             print('Best val Acc: {:4f}'.format(best_acc))
#
#             # 训练完后用最好的一次当做模型最终的结果
#             model.load_state_dict(best_model_wts)
#
#
#     return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs