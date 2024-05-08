# Author:司震
# Create: 2024/4/24
# Description: 第3天练习：频率域高斯低通滤波器
import cv2
import numpy as np
import matplotlib.pyplot as plt
def make_Gaussian_matrix(img,D0):
    (m,n)=img.shape
    u=np.array([i if i<=m/2 else m- i for i in range(m)],dtype=np.float32)
    v = np.array([i if i <= n / 2 else n - i for i in range(n)], dtype=np.float32)
    v.shape=n,1
    ret=np.fft.fftshift(np.sqrt(u*u+v+v))
    transfor_matrix=np.exp(-(ret*ret)/(2*D0*D0))
    return transfor_matrix.T
path=r"F:/ML_Python/Source/kobe.jpg"
img=cv2.imread(path,0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
D10=make_Gaussian_matrix(img,10)
new_10=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*D10)))
D20=make_Gaussian_matrix(img,20)
new_20=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*D20)))
D30=make_Gaussian_matrix(img,30)
new_30=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*D30)))
#滤波结果显示
plt.figure(dpi=400)
plt.subplot(221)
plt.imshow(img,'gray')
plt.title('original image')

plt.subplot(222)
plt.imshow(new_10,'gray')
plt.title('Frequency filter:D0=10')

plt.subplot(223)
plt.imshow(new_20,'gray')
plt.title('Frequency filter:D0=20')

plt.subplot(224)
plt.imshow(new_30,'gray')
plt.title('Frequency filter:D0=30')
plt.subplots_adjust(wspace=0,hspace=0.5)
plt.show()
