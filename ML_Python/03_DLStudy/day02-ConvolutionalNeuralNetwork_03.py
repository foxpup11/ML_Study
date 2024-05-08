# Author:司震
# Create: 2024/4/21
# Description: 第2天练习：函数式模型搭建卷积神经网络进行手写字符识别
import numpy as np
from keras.datasets import mnist
from keras.layers import Input,Conv2D,Dense,MaxPooling2D,Flatten,Dropout
from keras import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical #独立热编码
from keras.utils import plot_model #绘图
#分割数据集
x=mnist.load_data()[0][0]
y=mnist.load_data()[0][1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
x_train=x_train.reshape(36000,28,28,1)[:1000] #取出训练集前1000张图片用作训练
x_test=x_test.reshape(24000,28,28,1)[:500] #取出测试集前500张图片用作测试
x_train=np.array(x_train,dtype='float32')
y_train=np.array(y_train,dtype='float32')[:1000]#取出训练数据的前1000个标签
x_test=np.array(x_test,dtype='float32')
y_test=np.array(y_test,dtype='float32')[:500] #取出测试数据的前500个标签

#归一化
x_train=x_train/255
x_test=x_test/255
#对标签进行one-hot编码
y_train_new=to_categorical(y=y_train,num_classes=10)
y_test_new=to_categorical(y=y_test,num_classes=10)
#函数式搭建模型
mnist_input=Input(shape=(28,28,1))
conv2d=Conv2D(filters=30,kernel_size=(3,3),padding='valid',activation='relu')(mnist_input)
maxPooling=MaxPooling2D(pool_size=(2,2))(conv2d)
flatten=Flatten()(maxPooling)
dense=Dense(20,activation='relu')(flatten)
dropout=Dropout(0.5)(dense)
output=Dense(10,activation='softmax',name='output')(dropout)
model=Model(inputs=[mnist_input],outputs=[output])
#对模型进行初始化
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
#训练数据
model.fit(x_train,y_train_new,epochs=20,verbose=2,batch_size=64,validation_split=0.2)
#评价模型
print(model.evaluate(x_test,y_test_new))


