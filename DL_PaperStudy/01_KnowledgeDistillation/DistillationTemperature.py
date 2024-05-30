# Author:SiZhen
# Create: 2024/5/25
# Description: 可视化蒸馏温度
import numpy as np
import matplotlib.pyplot as plt

#输入各类别logits
logits = np.array([-5,2,7,9])

#普通softmax(T=1)
softmax_1 = np.exp(logits)/sum(np.exp(logits))
# print(softmax_1)
# plt.plot(softmax_1,label='softmax_1')
# plt.legend()
# plt.show()

#知识蒸馏softmax(T=3)
plt.plot(softmax_1,label='T=1')
T = 3
softmax_3 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_3,label='T=3')
T = 5
softmax_5 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_5,label='T=5')
T = 10
softmax_10 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_10,label='T=10')
T = 100
softmax_100 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_100,label='T=100')
plt.xticks(np.arange(4),['cat','Dog','Donkey','Horse'])
plt.legend()
plt.show()




