# python练习day01 2024/4/7
# print("hello world");print("this is my first program!!!!")
# python一行可以写多条语句用分号分开
# 多行注释ctrl+/
# a = input('please input the values:')#字符串类型
# b = eval(input ('please input the number:'))
# print(a)
# print('hello'+'world')

# 字符串练习
# s = "helloworld"
# print(len(s))
# print(s[::-1])
# print(s[::1])
# print(s[2:3:1])
# print(str(999))

# 列表练习
# list1=['你好','hello',57,[1,'hello',0.87]]
# print(list1)
# 字符串和列表的切片给定不包括终点，省略包括

# 元组练习
# tup1 = ('1',1,3.87,'wu')
# tup2 =(1,) #一个元素的元组后面的逗号不能省略
# tup3 =()
# print(1 in tup1)
# print('1' in tup1)
# print(tuple("hello"))
# print(tuple([1,2,'wu']))

# 字典练习
# dic = {'姓名':'李磊','性别':'男','数学':95}
# for key in dic:
#     print(key,":",dic[key])
# dic['数学']=99
# del dic['性别']
# dic['语文']=99
# for key in dic:
#     print(key,":",dic[key])

# 集合练习(无序不重复)
# basket= {"apple","banana","orange","apple","orange"}
# print(basket)
# basket.add("water")
# print(basket)

# 赋值练习
# a,b=10,20
# print(a,b)
# c,d = eval(input("please input the value:"))#输入要以逗号隔开，不能用空格
# print(c,d)
# e = 2**3;print(e)

# if单分支练习：求最大数
# a,b,c = eval(input("please input the three numbers:"))
# max_data = a
# if b>max_data:
#     max_data = b
# if c>max_data:
#     max_data = c
# print(max_data)
# print("the numbers are:{},{},{},the max data is:{}".format(a,b,c,max_data))

# 双分支练习：求解一元二次方程
# import math
# a,b,c =eval(input("input numbers:"))
# delt = b**2-4*a*c
# if delt>=0:
#     x1=(-b+math.sqrt(delt))/(2*a)
#     x2=(-b-math.sqrt(delt))/(2*a)
#     print("有两个根，分别为:{}和{}".format(x1,x2))
# else:
#     print('没有实根')

# 多分支练习
# a, b = 10, 20
# if a > b:
#     a = b
# elif b > a:
#     b = a

#while循环练习
# result=0
# i=1
# while i<=100:
#     result=result+i
#     i=i+1
# print(result)


#for循环，range函数练习
#py的for循环是元素遍历
#求100以内奇数和
# s =0
# for i in range(1,100,2):
#     s = s+i
# print(s)




