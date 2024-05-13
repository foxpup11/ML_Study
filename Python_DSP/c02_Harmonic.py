# Author:SiZhen
# Create: 2024/5/11
# Description:谐波
import thinkdsp as dsp
import matplotlib.pyplot as plt
import scipy

framerate = 10000
signal = dsp.CosSignal(4500)
duration=signal.period*5 #持续五个周期
segment = signal.make_wave(duration,framerate=framerate)
segment.plot()
plt.show()

signal = dsp.SinSignal(5500)
segment=signal.make_wave(duration,framerate=framerate)
segment.plot()
plt.show()