# Author:SiZhen
# Create: 2024/5/23
# Description: 离散傅里叶变换
import numpy as np
import matplotlib.pyplot as plt
import thinkdsp
from thinkdsp import decorate
PI2 = 2 * np.pi
from thinkdsp import Sinusoid
# suppress scientific notation for small numbers
np.set_printoptions(precision=3, suppress=True)


class ComplexSinusoid(Sinusoid):
    """Represents a complex exponential signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        print(ts)
        phases = PI2 * self.freq * ts + self.offset
        print(phases)
        ys = self.amp * np.exp(1j * phases)
        return ys
signal = ComplexSinusoid(freq=1, amp=0.6, offset=1)
wave = signal.make_wave(duration=1, framerate=4)
print(wave.ys)


from thinkdsp import SumSignal

def synthesize1(amps, freqs, ts):
    components = [ComplexSinusoid(freq, amp)
                  for amp, freq in zip(amps, freqs)]
    signal = SumSignal(*components)
    ys = signal.evaluate(ts)
    return ys


