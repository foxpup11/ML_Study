# Author:SiZhen
# Create: 2024/5/11
# Description: 非周期信号
import thinkdsp as dsp
import matplotlib.pyplot as plt
import scipy

#线性啁啾
signal = dsp.Chirp(start=220,end=880)
wave = signal.make_wave()
#指数啁啾
signal = dsp.ExpoChirp(start=220,end=880)
wave = signal.make_wave()

#创建一个1s八度啁啾信号，并计算频谱
signal = dsp.Chirp(start=220,end=440)
wave = signal.make_wave(duration=1)
spectrum = wave.make_spectrum()
# spectrum.plot()
# plt.show()

signal = dsp.Chirp(start=220,end=440)
wave = signal.make_wave(duration=1)
spectrogram=wave.make_spectrogram(seg_length=512)#seg_length为每个片段采样的数量
# spectrogram.plot(high=700)
# plt.show()