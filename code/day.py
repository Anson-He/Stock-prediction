import pandas as pd
import os
import re
import atrader as at
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 用CPU


# 获取上证A股代码
print('正在获取沪深300代码')
dirs = os.listdir("../data")


def get_name(s0):
    return re.findall('([0123456789]+).', s0)


target_list = []
for i in dirs:
    s = get_name(i)
    if len(s) != 0:
        if 'sse' in i:
            target = 'sse.'+s[0]
            target_list.append(target)
        if 'szse' in i:
            target = 'szse.' + s[0]
            target_list.append(target)
print(len(target_list))
for file_name in target_list:
    data = pd.read_csv('../data/'+file_name+'.csv', index_col=0)
    data['time'] = pd.to_datetime(data['time']).astype("datetime64[D]")
    data.set_index(data['time'], inplace=True)
    if int(str(data['time'][-1])[:4]) < 2021:
        print(file_name, "<2021")
        target_list.remove(file_name)
    if int(str(data['time'][0])[:4]) > 2011:
        print(file_name, ">2011")
        target_list.remove(file_name)
print('沪深300代码获取成功,总长度为', len(target_list))

# 加载数据


def load_data(file_path, fn):
    data0 = pd.read_csv(file_path, index_col=0)
    data0['time'] = pd.to_datetime(data0['time']).astype("datetime64[D]")
    # del data['Unnamed: 0']
    del data0['open_interest']
    data0.set_index(data0['time'], inplace=True)
    # del data['time']
    # yz = at.get_factor_by_code(factor_list=['PE', 'PB', 'PCF', 'PS', 'MKV', 'ARBR', 'BP'],
    # target=fn, begin_date=data0['time'][0], end_date=data0['time'][-1])
    # yz.set_index(yz['date'], inplace=True)
    # yz.drop(index=list(set(yz['date']) - set(data0['time'])))
    # data0['PE'] = yz['PE']
    # data0['PB'] = yz['PB']
    # data['PCF'] = yz['PCF']
    # data0['PS'] = yz['PS']
    # data0['MKV'] = yz['MKV']
    # data['ARBR'] = yz['ARBR']
    # data0['BP'] = yz['BP']
    return data0


for file_name in target_list:
    print('---------------'+file_name+'---------------------')
    # 读取数据
    data = load_data('../data/'+file_name+'.csv', file_name)
    print('数据读取成功')

    data = data[['close']]
    data = (data-data.min())/(data.max()-data.min())

    # 1.先开盘预测开盘
    x_train = np.array(data['2011':'2019']['close'][:-1])
    y_train = np.array(data['2011':'2019']['close'][1:])
    x_test = np.array(data['2020':]['close'][:-1])
    y_test = np.array(data['2020':]['close'][1:])

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # 样本数、时间步、特征
    x_train = x_train.reshape(x_train.shape[0], 1, 1)
    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, 1)
    y_test = y_test.reshape(y_test.shape[0], 1, 1)

    # 构建LSTM神经网络
    time_step = 1
    vector = 1
    output = 1

    model = Sequential()
    model.add(LSTM(units=128, input_shape=(time_step, vector), return_sequences=True))
    model.add(Dense(64))
    # model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(16))
    model.add(Dense(1))
    model.add(Activation("tanh"))
    print(model.summary())
    # model.compile(loss="mean_squared_error", optimizer='adam')
    #
    # print('----------------开始训练----------------------------')
    # history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))
    # print('----------------训练完成----------------------------')
    # scores = model.evaluate(x_test, y_test)
    # model.save('../model/day/'+file_name+'.h5')
    # print('模型保存完成')
    # # 绘制loss
    # plt.figure()
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend()
    # plt.title('Loss')
    # plt.savefig('../figure/day/loss_'+file_name+'.png')
    # plt.clf()  # 清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。
    # plt.close()  # 完全关闭图形窗口
    # # 预测结果与真实值比较
    # plt.figure()
    # predict_price = model.predict(x_test)
    # real_price = x_test
    # predict_price = predict_price.reshape(predict_price.shape[0],)
    # real_price = real_price.reshape(real_price.shape[0],)
    # plt.plot(real_price, color='blue', label='Real Price')
    # plt.plot(predict_price, color='red', label='Predicted Price')
    # plt.title('Price Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.savefig('../figure/day/pred_'+file_name+'.png')
    # plt.clf()  # 清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。
    # plt.close()  # 完全关闭图形窗口
