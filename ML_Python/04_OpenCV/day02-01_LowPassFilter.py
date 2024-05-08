# Author:司震
# Create: 2024/4/23
# Description: 第2天练习：低通/高通滤波器
import numpy as np
import cv2
path=r"F:/ML_Python/Source/kobe.jpg"
# image =cv2.imread(path)
# cv2.imshow('Original ',image)

#低通滤波器
#1.邻域均值滤波
# blurred=np.hstack([cv2.blur(image,(3,3)),cv2.blur(image,(5,5)),cv2.blur(image,(7,7))])
# cv2.imshow('Averaged',blurred)

#2.高斯滤波
# blurred=np.hstack([cv2.GaussianBlur(image,(3,3),4),cv2.GaussianBlur(image,(5,5),4),cv2.GaussianBlur(image,(7,7),4)])
# cv2.imshow('Gaussian',blurred)

#3.中值滤波
# blurred = np.hstack([cv2.medianBlur(image,5),cv2.medianBlur(image,7)])
# cv2.imshow("Median",blurred)

#高通滤波器
#以拉普拉斯算子为例
# image=cv2.imread(path,0)
# kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)#拉普拉斯算子
# dst = cv2.filter2D(image,-1,kernel=kernel)
# cv2.imshow('original',np.uint8(image))
# cv2.imshow('Shappern',np.uint8(dst))





key=cv2.waitKey(0)
if key==27: #按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()
