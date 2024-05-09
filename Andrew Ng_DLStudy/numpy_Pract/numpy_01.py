# Author:司震
# Create: 2024/5/6
# Description: 第1天练习：numpy库
import numpy as np;

#a = np.array([1,2,3])
#a = np.array([[1,2,3],[4,5,6]])
# a = np.array([1,2,3,4,5],ndmin=2)   #最小维度
# a= np.array([1,2,3],dtype=complex)#dtype参数，complex指复数（实数部分和虚数部分）
# print(a)

# dt = np.dtype(np.int32)
# dt = np.dtype('i4')#int8,int16,int32,int64 四种数据类型可以使用字符串'i1','i2','i4','i8'代替
#dt = np.dtype('<i4')
# dt = np.dtype([('age',np.int8)])
# a = np.array([(10,),(20,),(30,)],dtype=dt)
# print(dt)
# print(a['age'])

#定义一个结构化数据类型student
# student = np.dtype([('name','S20'),('age','i1'),('marks','f4')])
# a = np.array([('av',21,90),('xs',10,100)],dtype=student)
# print(a)

# a = np.array([[1,2,3],[4,5,6]])
# b = a.reshape(3,2)
# print(b)
# b[0][0]=100
# print(b)
# print(a) #可以发现，数组a内的值跟着改变了

# x = np.empty((3,2),dtype=np.int64) #该方法用来创建一个指定形状、数据类型且未初始化的数组
# print(x) #数组内的元素是随机的，并不一定是全0（有可能为全0）
# y= np.zeros((3,2),dtype=np.int64) #创建一个指定形状、数据类型且元素全0的数组
# print(y)

#自定义类型
# x = np.zeros([2,2],dtype=[('x','i8'),('y','i8')])
# print(x)

#创建全1数组
# x = np.ones([2,2],dtype=np.int64,order='C')#order代表在计算机内存中的存储顺序，‘C’代表行优先，'F'代表列优先，默认为‘K’（保留输入数组的存储顺序）
# print(x)

#默认为浮点数
# x = np.ones((5,))#一维的，不建议这么写
# print(x)
# print(x.ndim)
# y = np.ones((5,1))#二维数组，注意和上面区分
# print(y)
# print(y.ndim)
# z = np.ones_like(y,dtype=np.int64,shape=[2,2])
# print(z)

# list = range(5)
# it = iter(list)

# x=np.arange(5) #生成一个0~4的数组，start从0（默认值）开始，stop到5（不包含）结束，step步长为1（默认），dtype默认为输入数据的类型
# print(x)#最后返回ndarray的数据类型
# y = np.arange(5,dtype=float)
# print(y)
# z = np.arange(10,20,2,dtype=np.float32)
# print(z)

#linspace用于生成一个一维等差数列数组
# x = np.linspace(10,15,endpoint=True,num=50,retstep=True,dtype=np.float32)#start起始值，stop中止值，endpoint是否包含stop值（默认为true）,num生成等步长的样本数量（默认值50）
# print(x)#retstep是否显示步长值（默认为false）
# y = np.linspace(10,20,10,dtype=np.int64).reshape((10,1))#linspace和reshape结合使用
# print(y)

#logspace用于创建一个一维等比数列
# x = np.logspace(start=1,stop=6,num=6,base=2,dtype=np.float64) #该函数没有retstep参数,base参数意思是取对数的时候log的下标
# print(x)#start指base的start次方开始，stop指base的stop次方停止，endpoint默认为true，base是底数默认为10


# a=np.arange(10)
# print(a)
# # s =slice(2,7,2)#对ndarray格式的数组进行切片
# # print(a[s])
# b = a[2:7:2]#切片的另一种方法
# print(b)
# c = a[2:7]#step默认为1
# print(c)

# a=np.arange(25).reshape((5,5))
# print(a[...,0])#第1列元素
# print(a[0,...])#第一行元素
# print(a[...,...]) #不能这么写
# print(a)
# b=slice(2,5)#对多维数组进行切片
# print(a[b])

# x = np.arange(20).reshape(5,4)
# print(x)
# y = x[[1,0,3],[1,1,2]] #用数组索引数组，分别对应（1，1），（0，1），（3，2）
# print(y)

# x = np.arange(32).reshape(8,4)
# print(x)
# y = x[np.ix_([1,5,7,2],[0,3,1,2])]#产生笛卡尔积的映射关系
# print(y)

#


