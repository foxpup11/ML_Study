# Author:司震
# Create: 2024/4/21
# Description: 第1天练习：自动编码器
from keras.datasets import  mnist
from keras.models import Model
from keras.layers import Input,Dense
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from sklearn import manifold
import matplotlib.pyplot as plt


#采用keras构建自动编码器
#导入数据
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train = X_train.reshape(60000,28*28)
#数据规范化
X_train =X_train.astype('float32')/255
X_test = X_test.reshape(10000,28*28)
X_test=X_test.astype('float32')/255
scale=StandardScaler().fit(X_train)
X_train=scale.transform(X_train)
X_test=scale.transform(X_test)
encoding_dim = 2 #数据压缩到二维，方便显示
input_img = Input(shape=(784,)) #输入层的尺寸与图像中的像素个数相同
#搭建编码层
encoded=Dense(128,activation='relu')(input_img) #增加第一个编码层，包含128个结点
encoded=Dense(64,activation='relu')(encoded) #增加第二个编码层，包含64个结点
encoded=Dense(10,activation='relu')(encoded)#增加第三个编码层，含有10个结点
encoder_output=Dense(encoding_dim)(encoded) #增加第四个编码层，含有2个结点
#搭建解码层
decoded=Dense(10,activation='relu')(encoder_output) #增加第一个解码层，含有10个结点
decoded=Dense(64,activation='relu')(decoded) #增加第二个解码层，含有64个结点
decoded=Dense(128,activation='relu')(decoded) #增加第三个解码层，含有128个结点
decoded=Dense(784,activation='relu')(decoded) #增加第四个解码层，同时也是输出层
autoencoder=Model(inputs=input_img,outputs=decoded)#构建自动编码器
encoder=Model(inputs=input_img,outputs=encoder_output)#只包含编码层，用于数据压缩
autoencoder.compile(optimizer='adam',loss='mse')
autoencoder.fit(X_train,X_train,epochs=20,batch_size=256,shuffle=True)
encoded_imgs=encoder.predict(X_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test,s=3)
plt.colorbar()
plt.show()





