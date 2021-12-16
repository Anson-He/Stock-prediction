import pandas as pd
import os
import re
import atrader as at
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers


# 获取上证A股代码
print('正在获取上证A股代码')
dirs = os.listdir("../data")
print('上证A股代码获取成功')


def get_name(s0):
    return re.findall('([0123456789]+).', s0)


target_list = []
for i in dirs:
    s = get_name(i)
    if len(s) != 0:
        target = 'SSE.' + s[0]
        target_list.append(target)

# 加载数据


def load_data(file_path):
    data0 = pd.read_csv(file_path, index_col=0)
    data0['time'] = pd.to_datetime(data0['time']).astype("datetime64[D]")
    # del data['Unnamed: 0']
    del data0['open_interest']
    data0.set_index(data0['time'], inplace=True)
    # del data['time']
    yz = at.get_factor_by_code(factor_list=['PE','PB','PCF','PS','MKV','ARBR','BP'], target='SSE.600000', begin_date=data0['time'][0], end_date=data0['time'][-1])
    yz.set_index(yz['date'],inplace=True)
    yz.drop(index = list(set(yz['date']) - set(data0['time'])))
    data0['PE'] = yz['PE']
    data0['PB'] = yz['PB']
    # data['PCF'] = yz['PCF']
    data0['PS'] = yz['PS']
    data0['MKV'] = yz['MKV']
    # data['ARBR'] = yz['ARBR']
    data0['BP'] = yz['BP']
    return data0


# 读取数据
data = load_data('../data/SSE.600000.csv')
print('数据读取成功')

# 1.先开盘预测开盘
x_train = np.array(data['2011':'2018']['open'][:-1])
y_train = np.array(data['2011':'2018']['open'][1:])
x_test = np.array(data['2019':]['open'][:-1])
y_test = np.array(data['2019':]['open'][1:])

# 样本数、时间步、特征
x_train = x_train.reshape(1909, 1, 1)
y_train = y_train.reshape(1909, 1, 1)
x_test = x_test.reshape(649, 1, 1)
y_test = y_test.reshape(649, 1, 1)

# 构建LSTM神经网络
time_step = 1
vector = 1
output = 1

model = Sequential()
model.add(LSTM(units=32, input_shape=(time_step, vector), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("relu"))
model.compile(loss="mean_squared_error", optimizer='adam')

print('----------------开始训练----------------------------')
history = model.fit(x_train, y_train, epochs=60, batch_size=64, validation_data=(x_test, y_test))
print('----------------训练完成----------------------------')
scores = model.evaluate(x_test, y_test)
model.save('../model/day.h5')
print('模型保存完成')


# 绘制loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()


# 预测结果与真实值比较
predict_price = model.predict(x_test)
real_price = x_test
predict_price = predict_price.reshape(predict_price.shape[0],)
real_price = real_price.reshape(real_price.shape[0],)
plt.plot(real_price, color='blue', label='Real Price')
plt.plot(predict_price, color='red', label='Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

