# Author:SiZhen
# Create: 2024/5/13
# Description: 离散余弦变换
import numpy as np
PI2 = np.pi * 2
from thinkdsp import CosSignal, SumSignal

#信号合成
def synthesize1(amps, fs, ts):
    components = [CosSignal(freq, amp)
                  for amp, freq in zip(amps, fs)]
    signal = SumSignal(*components)

    ys = signal.evaluate(ts)
    return ys


from thinkdsp import Wave
amps = np.array([0.6, 0.25, 0.1, 0.05])
fs = [100, 200, 300, 400]
framerate = 11025

ts = np.linspace(0, 1, framerate, endpoint=False)
ys = synthesize1(amps, fs, ts)
wave = Wave(ys, ts, framerate)
wave.apodize()
wave.make_audio()
