# Author:SiZhen
# Create: 2024/5/20
# Description: 构建一个多层的神经网络实现猫分类器
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy

import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils

np.random.seed(1)

def initialize_parameters(n_x,n_h,n_y):
    """
       此函数是为了初始化两层网络参数而使用的函数。
       参数：
           n_x - 输入层节点数量
           n_h - 隐藏层节点数量
           n_y - 输出层节点数量

       返回：
           parameters - 包含你的参数的python字典：
               W1 - 权重矩阵,维度为（n_h，n_x）
               b1 - 偏向量，维度为（n_h，1）
               W2 - 权重矩阵，维度为（n_y，n_h）
               b2 - 偏向量，维度为（n_y，1）

       """
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    assert(W1.shape ==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape ==(n_y,1))
    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters

#初始化参数部分
def initialize_parameters_deep(layers_dims):
    """
        此函数是为了初始化多层网络参数而使用的函数。
        参数：
            layers_dims - 包含我们网络中每个图层的节点数量的列表

        返回：
            parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                         W1 - 权重矩阵，维度为（layers_dims [l]，layers_dims [l-1]）
                         bl - 偏向量，维度为（layers_dims [l]，1）
        """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])/np.sqrt(layers_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layers_dims[l],1))

        assert(parameters["W"+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape ==(layers_dims[l],1))
    return parameters

#前向传播-线性部分
def linear_forward(A,W,b):
    Z = np.dot(W,A) +b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)

    return Z,cache

#前向传播-线性激活部分
def linear_activation_forward(A_prev,W,b,activation):
    if activation == 'sigmoid':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)

    return A,cache

#多层模型的前向传播计算模型
def L_model_forward(X,parameters):
    """
        实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION

        参数：
            X - 数据，numpy数组，维度为（输入节点数量，样本数量）
            parameters - initialize_parameters_deep（）的输出

        返回：
            AL - 最后的激活值
            caches - 包含以下内容的缓存列表：
                     linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                     linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
        """
    caches=[]
    A = X
    L = len(parameters) //2 #//是整数除法
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL,caches

#计算成本
def compute_cost(AL,Y):
    """
       参数：
           AL - 与标签预测相对应的概率向量，维度为（1，样本数量）
           Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

       返回：
           cost - 交叉熵成本
    """
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m

    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

#反向传播-线性部分
def linear_backward(dZ,cache):
    """
        为单层实现反向传播的线性部分（第L层）

        参数：
             dZ - 相对于（当前第l层的）线性输出的成本梯度
             cache - 来自当前层前向传播的值的元组（A_prev，W，b）

        返回：
             dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
             dW - 相对于W（当前层l）的成本梯度，与W的维度相同
             db - 相对于b（当前层l）的成本梯度，与b维度相同
        """
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape ==W.shape)
    assert(db.shape ==b.shape)

    return dA_prev,dW,db

#反向的线性激活
def linear_activation_backward(dA,cache,activation='relu'):
    """
       实现LINEAR-> ACTIVATION层的后向传播。

       参数：
            dA - 当前层l的激活后的梯度值
            cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
            activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
       返回：
            dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
            dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
            db - 相对于b（当前层l）的成本梯度值，与b的维度相同
       """
    linear_cache,activation_cache =cache
    if activation =="relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation =='sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

#多层模型的反向传播函数
def L_model_backward(AL,Y,caches):
    """
        对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播

        参数：
         AL - 概率向量，正向传播的输出（L_model_forward（））
         Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
         caches - 包含以下内容的cache列表：
                     linear_activation_forward（"relu"）的cache，不包含输出层
                     linear_activation_forward（"sigmoid"）的cache

        返回：
         grads - 具有梯度值的字典
                  grads [“dA”+ str（l）] = ...
                  grads [“dW”+ str（l）] = ...
                  grads [“db”+ str（l）] = ...
        """
    grads={}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    for l  in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA"+str(l+1)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp

    return grads


#更新参数
def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) //2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters['b'+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters

#构建一个两层的神经网络
def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot = True):
    np.random.seed(1)
    grads = {}
    costs =[]
    (n_x,n_h,n_y) = layers_dims
    #初始化参数
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters['b1']
    W2=parameters["W2"]
    b2=parameters["b2"]
    #开始进行迭代
    for i in range(0,num_iterations):
        #前向传播
        A1,cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        #计算成本
        cost = compute_cost(A2,Y)
        #反向传播
        dA2 = -(np.divide (Y,A2)-np.divide(1-Y,1-A2))
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,"relu")

        #保存在grads
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        #更新参数
        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if i %100  == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第",i,"次迭代，成本值为：",np.squeeze(cost))

    #迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel('iterations(per tens)')
        plt.title("Learning rate ="+str(learning_rate))
        plt.show()

    return parameters


#加载数据集
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_x = train_x_flatten/255
train_y = train_set_y
test_x  = test_x_flatten/255
test_y = test_set_y

#数据集加载完成，训练两层模型
# n_x = 12288
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
# parameters = two_layer_model(train_x, train_set_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)

#进行预测
def predict(X,y,parameters):
    """
       该函数用于预测L层神经网络的结果，当然也包含两层

       参数：
        X - 测试集
        y - 标签
        parameters - 训练模型的参数

       返回：
        p - 给定数据集X的预测
       """
    m = X.shape[1]
    n = len(parameters)//2 #神经网络的层数
    p = np.zeros((1,m))

    #根据参数前向传播
    probas,caches = L_model_forward(X,parameters)

    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("准确度为："+str(float(np.sum((p==y)/m))))

    return p

# predictions_train = predict(train_x,train_y,parameters) #训练集
# predict_test = predict(test_x,test_y,parameters) #测试集

#搭建一个多层的神经网络
def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    """
       实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。

       参数：
           X - 输入的数据，维度为(n_x，例子数)
           Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
           layers_dims - 层数的向量，维度为(n_y,n_h,···,n_h,n_y)
           learning_rate - 学习率
           num_iterations - 迭代的次数
           print_cost - 是否打印成本值，每100次打印一次
           isPlot - 是否绘制出误差值的图谱

       返回：
        parameters - 模型学习的参数。 然后他们可以用来预测。
       """
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i  in range (0,num_iterations):
        AL,caches = L_model_forward(X,parameters)

        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)

        #打印成本值，如果print_cost=False则忽略
        if i %100 ==0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第",i,"次迭代，成本值为：",np.squeeze(cost))
    #绘图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations(per tens)")
        plt.title("Learning rate ="+str(learning_rate))
        plt.show()

    return parameters

#加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

#训练多层模型
layers_dims = [12288,20,7,5,1] #五层模型
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True,isPlot=True)

pred_train = predict(train_x, train_y, parameters) #训练集
pred_test = predict(test_x, test_y, parameters) #测试集

#查看识别错误的图片
def print_mislabeled_images(classes,X,y,p):
    a = p+y
    mislabeled_indices = np.asarray(np.where(a==1))
    plt.rcParams['figure.figsize'] = (40.0,40.0) #设置尺寸默认的大小
    nums_images = len(mislabeled_indices[0])
    for i  in range(nums_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2,nums_images,i+1)
        plt.imshow(X[:,index].reshape(64,64,3),interpolation='nearest')
        plt.axis('off')
        plt.title("prediction:"+classes[int(p[0,index])].decode("utf-8")+' \n Class:'+ classes[y[0,index]].decode("utf-8"))
    plt.show()

print_mislabeled_images(classes,test_x,test_y,pred_test)

#用我们自己的图片试试
# my_image = "dog.jpg"
# my_label_y = [1]
#
# fname = "datasets/"+my_image
# image = np.array(plt.imread(fname))
# my_image = scipy.misc.imresize(image,size=(64,64)).reshape((64*64*3,1))
# my_predicted_image = predict(my_image, my_label_y, parameters)
# plt.imshow(image)
# print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
#     int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
