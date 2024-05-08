# Author:司震
# Create: 2024/4/23
# Description: 第3天练习：巴特沃思低通滤波器
import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt
def butterworth_lpf(img,D0,n):
    blpf=np.zeros(img.shape)
    (r,c)=blpf.shape
    for u in range(r):
        for v in range(c):
            D=sqrt((u-r/2)**2+(v-c/2)**2)
            blpf[u,v]=1/(1+pow(D/D0,2*n))
    return blpf
path=r"F:/ML_Python/Source/kobe.jpg"
img=cv2.imread(path,0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
filter_mat1=butterworth_lpf(img,10,2)
filter_mat2=butterworth_lpf(img,20,2)
filter_mat3=butterworth_lpf(img,20,20)
blpf1=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*filter_mat1)))
blpf2=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*filter_mat2)))
blpf3=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*filter_mat3)))
#滤波结果显示
plt.figure(dpi=200)
plt.subplot(221)
plt.imshow(img,'gray')
plt.title('original image')
#以灰度形式显示图像
plt.subplot(222)
plt.imshow(blpf1,'gray')
plt.title('Frequency filter:D0=10,n=2')

plt.subplot(223)
plt.imshow(blpf2,'gray')
plt.title('Frequency filter:D0=20,n=2')

plt.subplot(224)
plt.imshow(blpf3,'gray')
plt.title('Frequency filter:D0=20,n=20')

plt.subplots_adjust(wspace=0,hspace=0.5)  #设置子图间距，避免遮挡
plt.show()