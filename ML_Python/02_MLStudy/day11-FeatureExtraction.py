# Author:司震
# Create: 2024/4/19
# Description: 第11天练习：特征提取
import cv2 as cv
import numpy as np
#使用OpenCV自带的基于haar-like 特征的人脸分类器
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv.imread('C:/Users/20807/Desktop/lena.jpg')
# gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # #人脸识别
# faces=face_cascade.detectMultiScale(gray,1.3,3)
# for (x,y,w,h) in faces:
#      #对每个人脸识别的结果用方框标记出来
#      cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# # # #识别结果显示
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.imwrite('recognize.jpg',img)
# cv.destroyAllWindows()

#读取图片，显示图片练习
# img = cv.imread("C:/Users/20807/Desktop/lena.jpg", cv.IMREAD_COLOR)  # 读取图片
# cv.imshow('input image', img)  # 显示图片
# cv.imwrite("test.jpg", img)  # 保存图片
# cv.waitKey()
#
# cv.destoryAllWindows()


#寻找视频中的绿色
# def extrace_object_demo():
#     capture = cv.VideoCapture("F:/20240416_203714.mp4")
#     while True:
#         ret, frame = capture.read()
#         if not ret:
#             break
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         lower_hsv = np.array([37, 43, 46])
#         upper_hsv = np.array([77, 255, 255])
#         mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
#         dst = cv.bitwise_and(frame, frame, mask=mask)
#         cv.imshow("video", frame)
#         cv.imshow("mask", dst)
#         c = cv.waitKey(40)
#         if c == 27:
#             break
# src = cv.imread("C:/Users/20807/Desktop/lena.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# extrace_object_demo()
# cv.waitKey()
#
# cv.destoryAllWindows()
