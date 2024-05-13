# Author:SiZhen
# Create: 2024/5/13
# Description:
import thinkdsp
import numpy as np
import matplotlib.pyplot as plt
from thinkdsp import decorate
from thinkdsp import SinSignal
import os



#定义一个函数创建不同相位差的正弦波
def make_sine(offset):
    signal = thinkdsp.SinSignal(freq=440,offset=offset)
    wave = signal.make_wave(duration=0.5,framerate=10000)
    return wave
#然后实例化具有不同相位差的两个波形
wave1 = make_sine(offset=0)
wave2 = make_sine(offset=1)

wave1.segment(duration=0.01).plot()
wave2.segment(duration=0.01).plot()
decorate(xlabel='Time (s)')
# plt.show()
#两个相互之间相位差为一个弧度的两个正弦波，其相关系数为0.54
# print(np.corrcoef(wave1.ys, wave2.ys))#相关矩阵，对角线元素是与其自身的相关性，恒为1，非对角线元素是wave1与wave2的相关性
# print(wave1.corr(wave2))

#序列相关性
def serial_corr(wave, lag=1):
    N = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:N-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    return corr

# 创建一个UG噪声，发现相关性是很弱的
from thinkdsp import UncorrelatedGaussianNoise
signal = UncorrelatedGaussianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
print(serial_corr(wave))

#创建一个布朗噪声，相关性是比较强的
from thinkdsp import BrownianNoise
signal = BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
print(serial_corr(wave))

#创建一个粉噪声，相关性应该介于UG和布朗噪声之间
from thinkdsp import PinkNoise
signal = PinkNoise(beta=1)
wave = signal.make_wave(duration=0.5, framerate=11025)
print(serial_corr(wave))



from thinkdsp import read_wave
wave = read_wave('28042__bcjordan__voicedownbew.wav')
wave.normalize()
spectrum = wave.make_spectrum()
#spectrum.plot()
#decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
#plt.show()

def plot_shifted(wave, offset=0.001, start=0.2):
    segment1 = wave.segment(start=start, duration=0.01)
    segment1.plot(linewidth=2, alpha=0.8)

    # start earlier and then shift times to line up
    segment2 = wave.segment(start=start-offset, duration=0.01)
    segment2.shift(offset)
    segment2.plot(linewidth=2, alpha=0.4)

    corr = segment1.corr(segment2)
    text = r'$\rho =$ %.2g' % corr
    plt.text(segment1.start+0.0005, -0.8, text)
    decorate(xlabel='Time (s)')

plot_shifted(wave, 0.0001)