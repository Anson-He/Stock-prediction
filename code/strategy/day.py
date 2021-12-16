from atrader import *
import pandas as pd
import numpy as np
import os
import re
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 用CPU

def init(context):
    # 注册每天行情
    reg_kdata(frequency='day', fre_num=1)
    set_backtest(initial_cash=1000000, stock_cost_fee=2, margin_rate=1.1)


def on_data(context):
    positions = context.account().position()
    print(positions)
    if positions is not None:
        print('卖出')
        order_close_all()  # 卖出平仓
    diff_dic = {}
    model_list = os.listdir('../../model/day')
    code_list = []
    for i in model_list:
        code_list.append(re.findall('(.{1,}).h5', i)[0])
    # 获取账户的资金情况
    cash = context.account().cash
    valid_cash = cash.valid_cash[0]
    df_day = get_reg_kdata(reg_idx=context.reg_kdata[0], df=True)
    ti = df_day['target_idx']
    t = df_day['time']
    oi = df_day['open_interest']
    del df_day['target_idx']
    del df_day['time']
    del df_day['open_interest']
    df_day = (df_day-df_day.min())/(df_day.max()-df_day.min())
    df_day['target_idx'] = ti
    df_day['time'] = t
    df_day['open_interest'] = oi
    for ind in range(len(model_list)):
        print(ind)
        model = load_model('../../model/day/' + model_list[ind])
        input = [list(df_day.close)[ind]]
        input = np.array(input)
        input = input.reshape(input.shape[0], 1, 1)
        diff_dic[code_list[ind]] = (model.predict(input)[0][0] - list(df_day.close)[ind])
    diff_list = sorted(diff_dic.items(), key=lambda item: item[1], reverse=True)
    close_list = []
    target_list = []
    for i in diff_list[:5]:
        target_list.append(i[0])
    for j in diff_list[:5]:
        close_list.append(j[1])
    s = sum(close_list)
    buy_list = []
    for ii in close_list:
        buy_list.append((ii / s))
    print('buy_list:', buy_list)
    # 策略下单交易：
    for ind in range(len(target_list)):
        order_percent(account_idx=0, target_idx=code_list.index(target_list[ind]), percent=float(buy_list[ind]), side=1, order_type=2, price=0.0, position_effect=1)
        # order_value(account_idx=0, target_idx=code_list.index(target_list[ind]), value=float(buy_list[ind]), side=1,
        #             order_type=4, price=0.0, position_effect=1)  # 买入下单
        print("买入")
    print(get_daily_orders())
    print(get_daily_executions())



if __name__ == '__main__':
    # 策略回测函数
    model_list = os.listdir('../../model/day')
    code_list = []
    for i in model_list:
        code_list.append(re.findall('(.{1,}).h5', i)[0])
    run_backtest(strategy_name='DL_day5', file_path='.', target_list=code_list,
                 frequency='day', fre_num=1, begin_date='2021-01-01', end_date='2021-10-01', fq=0)
