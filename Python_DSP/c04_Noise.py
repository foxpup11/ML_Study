# Author:SiZhen
# Create: 2024/5/12
# Description: 噪声
import thinkdsp as dsp
import scipy
import numpy as py
import matplotlib.pyplot as plt
import thinkplot
import thinkstats2

#产生0.5秒的UU噪声，每秒11025个样本
signal  =dsp.UncorrelatedUniformNoise()
wave = signal.make_wave(duration=0.5,framerate=11025)
#wave.play('UUnoise.wav') #把这个噪声保存起来听一听
# wave.plot()
# plt.xlim(0,0.1)#更改横轴的范围
# plt.show()

spectrum = wave.make_spectrum()#生成频谱
# spectrum.plot_power() #绘制能量
# plt.show()#横轴是频率，纵轴是功率

#绘制累计频谱
integ = spectrum.make_integrated_spectrum()
# integ.plot_power()
# plt.show()

#生成布朗噪声并绘制其波形
signal = dsp.BrownianNoise()
wave = signal.make_wave()
# wave.plot()
# plt.xlim(0,0.1)
# plt.show()
#查看布朗噪声的频谱
spectrum = wave.make_spectrum()
spectrum.plot_power(linewidth=1,alpha=0.5)
# thinkplot.config(xscale='log',yscale='log')
# plt.show()

#一般来说，白噪声是指不相关高斯白噪声（UG噪声）
#生成一个UG噪声的频谱，然后产生一个等概率图
signal = dsp.UncorrelatedGaussianNoise()
wave = signal.make_wave(duration=0.5,framerate=11025)
spectrum = wave.make_spectrum()
thinkstats2.NormalProbabilityPlot(spectrum.real)
plt.xlim(0,4)
plt.show()
thinkstats2.NormalProbabilityPlot(spectrum.imag)
plt.xlim(0,4)
plt.show()



