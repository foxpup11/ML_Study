# Author:SiZhen
# Create: 2024/5/11
# Description:声音与信号
import thinkdsp
from thinkdsp import decorate
import matplotlib.pyplot as plt
import scipy

#产生正弦信号
sin_sig = thinkdsp.SinSignal(freq=880,amp=0.5,offset=0)#offset为相位差，定义了信号周期的开始
#产生余弦信号
cos_sig = thinkdsp.CosSignal(freq=440,amp=1.0,offset=0)

mix = sin_sig+cos_sig #信号相加
#wave表示的是信号在一系列时间点下求出的值
wave = mix.make_wave(duration=0.5,start=0,framerate=11025) #长度，开始时间，每秒帧数
# wave.plot()
# plt.show()  #由于440hz的频率，绘制结果很像一个色块

period = mix.period #period是signal的属性，它返回的是信号的周期，单位为秒
segment = wave.segment(start=0,duration=period*3) #复制了前三个周期
# segment.plot()
# plt.show()

spectrum = wave.make_spectrum() #频谱
# spectrum.plot()
# plt.show()

spectrum.low_pass(cutoff=600,factor=0.01) #低通滤波器，将高于600的频率衰减为原来的1%
wave_1 = spectrum.make_wave()
# wave_1.play('temp.wav') #低通滤波器会使结果的声音比较压抑且昏暗

#生成三角波
signal = thinkdsp.TriangleSignal(200)
# signal.plot()
# plt.show()
wave_2 = signal.make_wave(duration=0.5,framerate=10000)#生成wave
spectrum=wave_2.make_spectrum() #生成频谱
# spectrum.plot()
# plt.show()




