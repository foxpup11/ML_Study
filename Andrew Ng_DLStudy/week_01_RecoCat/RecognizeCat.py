# Author:司震
# Create: 2024/5/9
# Description: 第X天练习：采用逻辑回归搭建一个识别猫的简单神经网络
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset() #获取数据集

# index = 25
# plt.imshow(train_set_x_orig[index])#对图片数据进行处理，不显示
# plt.show()#显示处理过的图片
# print("train_set_y"+str(train_set_y)) #显示训练集的标签

#使用np.squeeze的目的是压缩维度，只有压缩后的值才能进行解码操作
#print('y='+str(train_set_y[:,index])+",it's a "+classes[np.squeeze(train_set_y[:,index])].decode('utf-8')+"picture")

m_train = train_set_y.shape[1] #训练集里图片的数量
#print(train_set_y.shape)
m_test  = test_set_y.shape[1]  #测试集里图片的数量
num_px  = train_set_x_orig.shape[1] #图片的宽度和高度(均为64x64)
# test = train_set_x_orig.shape[0]
# print(test)
# print(m_train)

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #将训练集平铺成（64*64*3，209）的数组
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T  #将测试集...

train_set_x = train_set_x_flatten/255 #标准化，RGB的值/255可让所有数据落在[0,1]之间
test_set_x = test_set_x_flatten/255

#构建sigmod函数
def sigmod(z):
    s = 1/(1+np.exp(-z))
    return s

#初始化参数W和b
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim,1)) #参数初始化
    b = 0 #参数初始化
    #使用断言来确保我要的数据是正确的
    assert(w.shape==(dim,1)) #w的维度是（dim,1）
    assert(isinstance(b,float)or isinstance(b,int)) #b的类型是float或者是int

    return (w,b)

#实现前向和反向传播
def propagate(w,b,X,Y):
    m = X.shape[1]
    #前向传播
    A = sigmod(np.dot(w.T,X)+b) #计算激活值
    cost = (-1/m)* np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A))) #计算成本
    #反向传播
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    #使用断言确保数据正确
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost = np.squeeze(cost)
    assert(cost.shape==())

    #创建一个字典。把dw和db保存起来
    grads = {
        "dw":dw,
        "db":db
    }
    return(grads,cost)

#测试前向反向传播
# w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
# grads,cost = propagate(w,b,X,Y)
# print(grads['dw'])
# print(grads['db'])
# print(cost)

#通过梯度下降算法来优化w和b
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        #记录成本
        if i %100 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i%100==0):
            print("迭代的次数：%i,误差值：%f"%(i,cost))
    params={
        'w':w,
        'b':b
    }
    grads={
        "dw":dw,
        "db":db
    }
    return(params,grads,costs)

#预测标签是0还是1
def predict(w,b,X):
    m = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    #预测猫在图片中出现的概率
    A = sigmod(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        #将概率a[0,i]转换为实际预测p[0,i]
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0
    assert(Y_prediction.shape==(1,m))
    return Y_prediction

#测试预测效果
# w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
# print(predict(w,b,X))

#将之前做的所有工作整合到一个模型里面
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    w,b = initialize_with_zeros(X_train.shape[0]) #初始化参数
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)#优化w,b,拿到dw,db和cost

    #从字典参数中检索参数w和b
    w,b = parameters["w"],parameters["b"]

    #预测测试/训练集的例子
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    #打印训练后的准确性
    print("训练集准确性：",format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100),'%')
    print("测试集准确性：",format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100),'%')

    d ={
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_test,
        "w":w,
        'b':b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations
    }
    return d

#测试模型
d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.004,print_cost=True)

#画图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title('Learning rate = '+str(d["learning_rate"]))
plt.show()





