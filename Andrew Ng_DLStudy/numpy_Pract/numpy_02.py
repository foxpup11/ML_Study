# Author:司震
# Create: 2024/5/7
# Description: 第2天练习
import numpy as np

# a =np.arange(6).reshape(2,3)
# print(a)
# for x in np.nditer(a):#迭代输出数组
#      print(x)
# for x in np.nditer(a.T):
#     print(x)
#迭代a和a的转置的输出是一样的

# for x in np.nditer(a,order='F'): #列优先
#     print(x)
#
# print('\n')
# for x in np.nditer(a,order='C'): #行优先
#     print(x)

# a = np.arange(8).reshape(4,2)
# print(a)
# b = a.flatten(order='C') #数组扁平化，返回一维数组
# c = a.flatten(order='F') #按列优先进行数组扁平化
# print(b)
# print(c)

#numpy.revel()作用为展平数组元素，但与flatten不同的是，revel返回的是数组视图

# a = np.arange(8).reshape(2,4)
# print(a)
# b = a.transpose() #互换数组的维度(与.T类似)
# print(b)
# c = a.T
# print(c)

# x = np.arange(16).reshape(1,4,4)
# y = np.squeeze(x) #从给定数组形状中删除一个维度的条目
# print(x.shape)
# print(y.shape)

# x =np.arange(16).reshape(4,4)
# y = np.arange(16,32).reshape(4,4)
# print(x)
# print('\n')
# print(y)
# print('\n')
# a = np.concatenate((x,y),axis=0)#按列拼接，0轴
# b = np.concatenate((x,y),axis=1)#按行拼接，1轴
# a = np.stack((x,y),axis=0)
# b = np.stack((x,y),axis=1)
# print(a)
# print('\n')
# print(b)

#np.hstack()水平堆叠数组,np.vstack()垂直堆叠数组

# a = np.arange(9).reshape(3,3)
# print(a)
# b = np.split(a,3)#后面跟的如果是一个数，就用该数平均切分
# c = np.split(a,[0,2])#如果是一个数组，为沿轴切分的位置（左开右闭）,从中间切
# print(b)
# print(c)

#np.hsplit用于水平分割数组，np.vsplit用于垂直分割数组

# a =np.arange(8).reshape(4,2)
# print(a)
# # a.reshape(2,4)#对a本身不改变
# # print(a)
# b = np.resize(a,(3,4))#如果新数组大小大于原始大小，则包含原始数组中的元素的副本
# # print(b)
# print('\n')