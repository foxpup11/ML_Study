# Author:司震
# Create: 2024/4/22
# Description: 第1天练习：opencv基础
import cv2
import matplotlib.pyplot as plt
import numpy as np
#path=r"F:\ML_Python\Source\kobe.jpg" # 正反斜线都可以
path=r"F:/ML_Python/Source/kobe.jpg"
#img=cv2.imread(path)
#,cols=img.shape[:2]
#图片灰度处理
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#将图片转为灰度图
#cv2.imshow("manba out!",gray)#生成灰度图
#cv2.imshow("manba out!",img)#生成原图

#图片平移
# H = np.float32([[1,0,50],[0,1,25]]) #构造平移矩阵，X方向移动50，Y方向移动25
# rows,cols=img.shape[:2]
# res=cv2.warpAffine(img,H,(cols,rows)) #注意这里rows和cols需要反置
# cv2.imshow('origin_picture',img)
# cv2.imshow('new_picture',res)

#图片缩放
#设置X方向和Y方向的缩放因子
# res1=cv2.resize(img,None,fx=0.8,fy=0.5,interpolation=cv2.INTER_LINEAR)
# height,width=img.shape[:2]
#二是直接设置图像的大小，不需要缩放因子
# res2=cv2.resize(img,(int(0.8*width),int(0.8*height)),interpolation=cv2.INTER_AREA)
# cv2.imshow('manba',img)
# cv2.imshow('manba01',res1)
# cv2.imshow('manba02',res2)
# cv2.waitKey(0)

#图像旋转
# rows,cols=img.shape[:2]
#第一个参数是旋转中心，第二个参数是旋转角度，第三个参数是缩放比例
# M1=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.5)
# M2=cv2.getRotationMatrix2D((cols/2,rows/2),45,2)
# M3=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
# res1=cv2.warpAffine(img,M1,(cols,rows))
# res2=cv2.warpAffine(img,M2,(cols,rows))
# res3=cv2.warpAffine(img,M3,(cols,rows))
# cv2.imshow('res1',res1)
# cv2.imshow('res2',res2)
# cv2.imshow('res3',res3)


#图像仿射变换
# pts1=np.float32([[50,50],[200,50],[50,200]])
# pts2=np.float32([[10,100],[200,50],[100,250]])
#类似于构造矩阵
# M=cv2.getAffineTransform(pts1,pts2)
# res=cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('original',img)
# cv2.imshow('res',res)

#图像镜像翻转
# img_info=img.shape
# image_height=img_info[0]
# image_weight=img_info[1]
# cv2.imshow('image_arc',img)
# dst=np.zeros(img.shape,np.uint8)
# for i in range(image_height):
#     for j in range(image_weight):
#         dst[i,j]=img[image_height-i-1,j]
# cv2.imshow('image_dst',dst)

#图像二值化(全局阈值)
# img =cv2.imread(path,0) #必须为灰度图，单通道
# ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# titles=['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images=[img,thresh1,thresh2,thresh3,thresh4,thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

# 图像二值化(自适应阈值)
# img=cv2.imread(path,0)
# ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY) #全局阈值
# th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)#通过中位数方法计算阈值
# th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#通过高斯方法计算阈值
# titles=['Original Image','Global Thresholding(V=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
# images=[img,th1,th2,th3]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

#图像的线性变换,本例把原始图像中灰度乘1.5后加50
# img =cv2.imread(path)
# grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# height,width=grayImage.shape[:2]
# result=np.zeros((height,width),np.uint8) #图像灰度上移变换
# for i in range(height):
#     for j in range(width):
#         if int (grayImage[i,j]*1.5+50)>255:
#             gray=255
#         else:
#             gray=grayImage[i,j]*1.5+50
#         result[i,j]=np.uint8(gray)
# cv2.imshow('src',grayImage)
# cv2.imshow('result',result)

#图像的指数变换（伽马变换）
# img=cv2.imread(path)
# grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
# height,width=grayImage.shape[:2]
# result= np.zeros((height,width),np.uint8)
# c,y=1,0.8
# for i in range(height):
#     for j in range(width):
#         gray = c* grayImage[i,j]**y #伽马变换计算公式
#         if grayImage[i,j]>255:
#             gray=255
#         result[i,j]=np.uint8(gray)
# cv2.imshow('src',grayImage)
# cv2.imshow('result',result)

#直方图均衡化，提高图像对比度
# img = cv2.imread(path,0)
# img1=cv2.equalizeHist(img)
# cv2.imshow('01',img)
# cv2.imshow('02',img1)



















key=cv2.waitKey(0)
if key==27: #按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()


