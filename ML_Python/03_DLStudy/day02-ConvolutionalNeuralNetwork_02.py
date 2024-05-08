# Author:司震
# Create: 2024/4/21
# Description: 第2天练习：卷积神经网络02：用序贯式（sequetial）模型搭建法来搭建卷积神经网络进行手写字符识别
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.utils import  to_categorical
#用序贯式（sequetial）模型搭建法来搭建卷积神经网络进行手写字符识别
x=mnist.load_data()[0][0]#训练数据
y=mnist.load_data()[0][1]#标签
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
x_train=x_train.reshape(36000,28,28,1)[:1000]#取出训练集数据前1000张照片用作训练
x_test=x_test.reshape(24000,28,28,1)[:500]#取出测试集前500张用作测试
x_train=np.array(x_train,dtype='float32')
y_train=np.array(y_train,dtype='float32')[:1000]#取出与训练数据对应的前1000个标签
x_test=np.array(x_test,dtype='float32')
y_test=np.array(y_test,dtype='float32')[:500]#取出与测试数据对应的前500个标签
#归一化
x_train=x_train/255
x_test=x_test/255
#对标签进行one_hot编码
y_train_new=to_categorical(y=y_train,num_classes=10)
y_test_new=to_categorical(y=y_test,num_classes=10)
#搭建模型
model = Sequential()
model.add(Conv2D(filters=30,kernel_size=(3,3),padding="same",activation='relu',input_shape=[28,28,1]))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())#扁平化操作
#搭建全连接层，本层有20个神经元，激活函数为Relu
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.5)) #随即丢弃50%的数据，防止过拟合化
#mnist手写数据集有10类（0-9），因此最后一层全连接层使用10个神经元，并选择与其配套的激活函数softmax
model.add(Dense(10,activation='softmax'))
#模型参数配置
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc']) #选择adam优化器，使用交叉熵损失函数,使用acc评估模型的结果
#模型训练
model.fit(x_train
          ,y_train_new #经过one-hot编码后的标签
          ,epochs=20,verbose=2 #训练过程可视化
          ,batch_size=64
          ,validation_split=0.2)
#评价模型
print(model.evaluate(x_test,y_test_new))

