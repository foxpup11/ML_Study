#2024/4/9 练习day03
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='SimHei'

#绘图
# x = np.linspace(0,10,20) #生成x坐标,创建0-10之间等步长20个样本点数量
# y1 = x**2
# y2 = -x**2+5*x-6
# y3 = -2*x -10
# plt.plot(x,y1,'r--o',x,y2,'b-+',x,y3,'y:h')#画图
# plt.legend(['y=x**2','y=-x**2+5*x-6','y=-2*x-10'])#为坐标图添加图例
# plt.grid(True)#为坐标图添加网格
# plt.show()

#图形美化
# x=np.linspace(0,2*np.pi,100)
# y = np.sin(x)
# plt.plot(x,y)
# plt.xlabel('this is axis x ')
# plt.ylabel('this is axis y')
# plt.title('this is an example')
# plt.text(3,0,'mark here')
# plt.grid(True)
#plt.show()
#plt.savefig(,dpi=600)

#绘制子图
# x = np.linspace(0,10,50)
# plt.figure(figsize=(16,12))
# plt.subplot(2,3,1)
# plt.plot(x,np.exp(x))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('y=exp(x)')
# plt.show()

#设置中文
# matplotlib.rcParams['font.family']='SimHei'
# x = np.linspace(0,10,50)
# y = x**2-2*x+10
# plt.plot(x,y)
# plt.xlabel('x坐标',fontproperties='SimHei', fontsize=20)
# #plt.xlabel('x坐标')
# plt.ylabel('y坐标')
# plt.title('显示中文')
# plt.show()

#饼状图
# label =['教授','副教授','讲师','助教']
# size =[10,40,35,15]
# explode = [0,0,0.1,0]
# plt.pie(size,labels=label,explode=explode,shadow=True,autopct='%0.1f%%')
# plt.axis('equal')
# plt.legend
# plt.show()

#条形图
#..

#散点图
# plt.rcParams['axes.unicode_minus']=False
# plt.figure(figsize=(10,6))
# x1 = np.random.randn(10)
# x2 = np.random.randn(10)
# y1 = np.random.randn(10)
# y2 = np.random.randn(10)
# plt.scatter(x1,y1,marker='o',color='red',s=5,linewidths=25,alpha=0.2)
# plt.scatter(x2,y2,marker='+',color='blue',s=100,linewidths=58,alpha=0.7)
# plt.legend(['boy,girl'])
# plt.show()