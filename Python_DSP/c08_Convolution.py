# Author:SiZhen
# Create: 2024/5/24
# Description: 滤波与卷积
import numpy as np
import matplotlib.pyplot as plt

from thinkdsp import decorate
# suppress scientific notation for small numbers
np.set_printoptions(precision=3, suppress=True)
import pandas as pd

df = pd.read_csv('FB_2.csv', header=0, parse_dates=[0])
print(df.head())
print(df.tail())


