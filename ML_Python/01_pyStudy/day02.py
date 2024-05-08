# python练习day02 2024/4/8
#numpy练习
import numpy as np
import matplotlib.pyplot as plt

#求和：矩阵a平方和和矩阵b的立方
# a = [[12,7,3],[4,5,6],[7,8,9]]
# b = [[5,8,1],[6,7,3],[4,5,9]]
# a =np.array(a)
# b = np.array(b)
# result = a**2+b**3
# print(result)

#创建数组
# a=np.array([1,1,1],[2,2,2],(1,2,3))#列表和元组混搭创建二维数组
# b = np.array([1,2,3,4,5],dtype=np.int32)#指定数组中的数据为32位整型

#生成全0数组
# a = np.array([[1,2],[3,4],[5,6]])
# b = np.zeros([3,3])#生成3x3零矩阵
# c = np.zeros_like(a)#生成与矩阵a同型的零矩阵
# print(b)
# print(c)

#生成单位矩阵
# a = np.eye(3)
# print(a)

#生成全为某个数的矩阵
# a = np.full([3,3],9)
# print(a)
# b = np.array([[1,2],[3,4],[5,6]])
# c = np.full_like(b,9)
# print(c)

#生成等差数组
# a = np.arange(2,15)#默认是一维数组
# # print(a)
# # b = np.arange(48).reshape([6,8])#改变矩阵的行列
# # print(b)

#创建线性一维数组
# x = np.linspace(0,2*np.pi,100)
# plt.plot(x,np.sin(x),'r-o',x,np.cos(x),'b-.')
# plt.legend(['y=sin(x)','y=cos(x)'])
# plt.grid(True)
# plt.show()

#在[0,1）之间等概率生成1000个点并显示出来
# n = 1000
# x = np.random.uniform(0,1,n)
# y = np.random.uniform(0,1,n)
# plt.scatter(x,y)
# plt.axis('equal')
# plt.show()

#生成1000个均值为0，方差为1的符合正态分布的点
# n= 1000
# x = np.random.normal(0,1,n)
# y = np.random.normal(0,1,n)
# plt.scatter(x,y)
# plt.axis('equal')
# plt.show()

#数组练习
# a = np.arange(18).reshape([3,6])#创建一个三行六列值为0-17的矩阵
# print(a)
#print(a+10)#数组所有元素与10相加
# b = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
# print(a+b)

#数组扁平化
# a = np.arange(18).reshape([3,6])
# print(a)
# b = a.flatten(order = 'c')#按行扁平化为一维数组
# print(b)
# c = a.flatten(order = 'f')#按列扁平化一维数组
# print(c)

#数组转为列表
# a = np.arange(18).reshape([3,6])
# b = a.tolist
# print(a)
# print(b)

#数组排序
# np.random.seed(100)
# data = np.random.randint(1,100,(4,6))
# print(data)
# data1 = np.sort(data,axis=0)
# print(data1)
# data2 = np.sort(data,axis=1)
# print(data2)

#数组的索引和切片
# a = np.random.randint(1,100,[5,5])
# print(a)
# print(a[1:3])#选择第2，3行
# print(a[:,1:3])#选择第2，3列
# print(a[:,::2])#按步长为2选择列
# print(a[1:3,1:-1:2])#选择第2，3行，步长为2的第2-最后一列
#print(a[:2,[2,3,2]])#选择第1，2行，1，2，1列的数据（第1列取了两次）

#单个条件索引
# a = np.random.randint(1,100,[5,5])
# print(a)
# print(a[a>50])

#蒙特卡罗法计算圆周率
# n = 100
# 000
# x = np.random.rand(n)
# y = np.random.rand(n)
# circle_in = x[x**2+y**2<=1]
# print(circle_in.size/n*4)
# plt.scatter(x[x**2+y**2<=1],y[x**2+y**2<=1])
# plt.scatter(x[x**2+y**2>1],y[x**2+y**2>1])
# plt.axis('equal')
# plt.show()

