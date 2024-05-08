# Author:司震
# Create: 2024/4/23
# Description: 第2天练习：理想低通滤波器模板
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
path=r"F:/ML_Python/Source/kobe.jpg"
img=cv2.imread(path,0)
#fft库完成从空间域到频率域的转换
f=np.fft.fft2(img)
#频率中心的移动
fshift=np.fft.fftshift(f)
#定义滤波器模板
def make_transfrom_matrix(d,image):
    transfor_matrix=np.zeros(image.shape)
    center_x=(image.shape[0]-1)/2
    center_y=(image.shape[0]-1)/2
    for i in range(transfor_matrix.shape[0]):
        for j  in range(transfor_matrix.shape[1]):
            dis = sqrt((i-center_x)**2+(j-center_y)**2)
            if dis<d:
                transfor_matrix[i,j]=1
    return  transfor_matrix
#频率域滤波
D10=make_transfrom_matrix(10,img)
new_10=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*D10)))
D20=make_transfrom_matrix(20,img)
new_20=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*D20)))
D30=make_transfrom_matrix(30,img)
new_30=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*D30)))
#滤波结果展示
plt.subplot(221)
plt.imshow(img,'gray') #以灰度形式显示图像
plt.title('original image')
plt.subplot(222)
plt.imshow(new_10,'gray')
plt.title('Frequency filter:D0=10')
plt.subplot(223)
plt.imshow(new_20,'gray')
plt.title('Frequency filter:D0=20')
plt.subplot(224)
plt.imshow(new_30,'gray')
plt.title('Frequency filter:D0=20')
plt.subplots_adjust(wspace=0,hspace=0.5) #设置子图间距，避免遮挡
plt.show()


key=cv2.waitKey(0)
if key==27: #按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()