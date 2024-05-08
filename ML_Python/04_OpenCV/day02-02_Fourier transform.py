# Author:司震
# Create: 2024/4/23
# Description: 第2天练习：利用傅里叶变换显示频谱图象
import cv2
import numpy as np
import matplotlib.pyplot as plt
path=r"F:/ML_Python/Source/kobe.jpg"
img=cv2.imread(path,0)
#fft库从空间域到频率域的转换
f=np.fft.fft2(img)  #将图像img做离散傅里叶变换，f为img对应的频率域结果
fshift=np.fft.fftshift(f)  #将低频移动到中心
#取绝对值：将复数变化成实数
#取对数的目的是将数据范围压缩，能正常显示，否则频率图像中低频数据极大，而高频数据极小
s1=np.log(np.abs(fshift))
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('original')
plt.subplot(122)
plt.imshow(s1,'gray')
plt.title('Frequency Domain')
plt.show()


