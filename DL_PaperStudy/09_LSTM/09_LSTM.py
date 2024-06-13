# Author:SiZhen
# Create: 2024/6/7
# Description: pytorch实现LSTM用于股票预测+时间序列预测
#代码参考自https://blog.csdn.net/qq_44824148/article/details/126222872?spm=1001.2014.3001.5506
#对收盘价(Close)单特征进行预测，利用前n天的数据预测第n+1天的数据。

#1.导入数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler

filepath = r'F:\ML_practice\DL_PaperStudy\09_LSTM\rlData.csv'
data = pd.read_csv(filepath)
#将数据按照日期进行排序，确保时间序列递增
data = data.sort_values('Date')

#2.将股票数据收盘价（Close）进行可视化展示
#设置画布大小
plt.figure(figsize=(15,9))
plt.plot(data[['Close']])
plt.xticks(range(0,data.shape[0],20),data['Date'].loc[::20],rotation=45)
plt.title("Stock Price(USD)",fontsize=18,fontweight='bold')
plt.xlabel("Date",fontsize=18)
plt.ylabel('Close Price(USD)',fontsize=18)
#plt.savefig('StockPrice.jpg')
plt.show()

#3.特征工程
#选取Close作为特征
price = data[['Close']]
#print(price.info())

#归一化操作
scaler = MinMaxScaler(feature_range=(-1,1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
#print(price['Close'].shape)

#4.制作数据集
#今天的收盘价预测明天的收盘价
#lookback表示视察的跨度
def split_data(stock,lookback):
    #将stock转化为ndarray类型
    data_raw = stock.to_numpy()
    data = []
    #将data按lookback分组，data为长度为lookback的list
    for index in range(len(data_raw)-lookback):
        data.append(data_raw[index:index+lookback])

    data = np.array(data)

    #按照8：2进行训练集、测试集划分
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size,:-1,:]
    y_trian = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    return [x_train,y_trian,x_test,y_test]

lookback =20
x_train,y_train,x_test,y_test = split_data(price,lookback)

#5.模型构建
import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
#真实的数据
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

#输入的维度为1，只有Close收盘价
input_dim = 1
#隐藏层特征的维度
hidden_dim = 32
#循环的layers
num_layers = 2
#预测后一天的收盘价
output_dim = 1

num_epochs = 100

class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()
        out,(hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
        out = self.fc(out[:,-1,:])
        return out

model = LSTM(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim,num_layers=num_layers)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(),lr=0.01)

#6.模型训练
import time

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred,y_train_lstm)
    print ("Epoch",t,"MSE:",loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time() -start_time
print("Training time:{}".format(training_time))

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
print(predict) #预测值
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
print(original) #真实值

#7.模型结果可视化
sns.set_style("darkgrid")

fig = plt.figure()
fig.subplots_adjust(hspace=0.2,wspace=0.2)

plt.subplot(1,2,1)
ax = sns.lineplot(x = original.index, y = original[0],label='Data',color='royalblue')
ax = sns.lineplot(x = predict.index,y = predict[0],label = 'Training Prediction(LSTM)',color='tomato')
print(predict.index)
print("aaaa")
print(predict[0])

ax.set_title('Stock price',size= 14,fontweight = 'bold')
ax.set_xlabel("Days",size=14)
ax.set_ylabel('Cost(USD)',size=14)
ax.set_xticklabels('',size=10)

plt.subplot(1,2,2)
ax = sns.lineplot(data=hist,color = 'royalblue')
ax.set_xlabel("Epoch",size = 14)
ax.set_ylabel("Loss",size = 14)
ax.set_title("Training Loss",size = 14,fontweight = 'bold')
fig.set_figheight(6)
fig.set_figwidth(16)
plt.show()

#8.模型验证
import math, time
from sklearn.metrics import mean_squared_error

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred) + lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred) + lookback - 1:len(price) - 1, :] = y_test_pred

original = scaler.inverse_transform(price['Close'].values.reshape(-1, 1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                                    mode='lines',
                                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                         mode='lines',
                         name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                                    mode='lines',
                                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Close (USD)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template='plotly_dark'

)

annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                        xanchor='left', yanchor='bottom',
                        text='Results (LSTM)',
                        font=dict(family='Rockwell',
                                  size=26,
                                  color='white'),
                        showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()
