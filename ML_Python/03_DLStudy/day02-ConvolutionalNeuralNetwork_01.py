# Author:司震
# Create: 2024/4/21
# Description: 第2天练习：卷积神经网络01
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.family']='SimHei'
#卷积过程的代码
img = plt.imread(r"F:/ML_Python/Source/tomato.jpg") #读图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #彩色图像转为灰度图像, 注意：cvtColor中间没有.
#以下定义了五种卷积模板
sobel_x=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
sobel_y=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
prewitt_x=np.array(([-1,0,1],[-1,0,1],[-1,0,1]))
prewitt_y=np.array(([-1,-1,-1],[0,0,0],[1,1,1]))
laplacian=np.array(([0,-1,0],[-1,4,-1],[0,-1,0]))
#以下采用上面五种卷积模板分别与图像进行卷积
im_sobel_x=cv2.filter2D(gray,-1,sobel_x)#卷积函数
im_sobel_y=cv2.filter2D(gray,-1,sobel_y)
im_prewitt_x=cv2.filter2D(gray,-1,prewitt_x)
im_prewitt_y=cv2.filter2D(gray,-1,prewitt_y)
im_laplacian=cv2.filter2D(gray,-1,laplacian)
#显示卷积后的图像
plt.subplot(2,3,1)
plt.imshow(gray,'gray')
plt.title('原始图像')

plt.subplot(2,3,2)
plt.imshow(im_sobel_x,'gray')
plt.title('im_sobel_x')

plt.subplot(2,3,3)
plt.imshow(im_sobel_y,'gray')
plt.title('im_sobel_y')

plt.subplot(2,3,4)
plt.imshow(im_prewitt_x,'gray')
plt.title('im_prewitt_x')

plt.subplot(2,3,5)
plt.imshow(im_prewitt_y,'gray')
plt.title('im_prewitt_y')

plt.subplot(2,3,6)
plt.imshow(im_laplacian,'gray')
plt.title("im_laplacian")

plt.subplots_adjust(wspace=0.5,hspace=0.5) #设置子图之间的距离
plt.show()




