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
    reg_factor(factor=['PE', 'PB', 'PS', 'MKV', 'BP'])
    set_backtest(initial_cash=1000000, stock_cost_fee=2, margin_rate=1.1)


def on_data(context):
    if int(context.now.strftime("%Y-%m-%d")[5:7]) >= 9:
        # data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=7, df=True)
        # print(context.now.strftime("%Y-%m-%d"))
        positions = context.account().position()
        print(positions)
        if positions is not None:
            print('卖出')
            order_close_all()  # 卖出平仓
        YZ = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=7, df=True)
        PEs = YZ[YZ['factor'] == 'PE']
        ti = PEs['target_idx']
        date = PEs['date']
        factor = PEs['factor']
        del PEs['target_idx']
        del PEs['date']
        del PEs['factor']
        PEs = (PEs - PEs.min()) / (PEs.max() - PEs.min())
        PEs['target_idx'] = ti
        PEs['date'] = date
        PEs['factor']= factor

        PBs = YZ[YZ['factor'] == 'PB']
        ti = PBs['target_idx']
        date = PBs['date']
        factor = PBs['factor']
        del PBs['target_idx']
        del PBs['date']
        del PBs['factor']
        PBs = (PBs - PBs.min()) / (PBs.max() - PBs.min())
        PBs['target_idx'] = ti
        PBs['date'] = date
        PBs['factor'] = factor

        PSs = YZ[YZ['factor'] == 'PS']
        ti = PSs['target_idx']
        date = PSs['date']
        factor = PSs['factor']
        del PSs['target_idx']
        del PSs['date']
        del PSs['factor']
        PSs = (PSs - PSs.min()) / (PSs.max() - PSs.min())
        PSs['target_idx'] = ti
        PSs['date'] = date
        PSs['factor'] = factor

        MKVs = YZ[YZ['factor'] == 'MKV']
        ti = MKVs['target_idx']
        date = MKVs['date']
        factor = MKVs['factor']
        del MKVs['target_idx']
        del MKVs['date']
        del MKVs['factor']
        MKVs = (MKVs - MKVs.min()) / (MKVs.max() - MKVs.min())
        MKVs['target_idx'] = ti
        MKVs['date'] = date
        MKVs['factor'] = factor

        BPs = YZ[YZ['factor'] == 'BP']
        ti = BPs['target_idx']
        date = BPs['date']
        factor = BPs['factor']
        del BPs['target_idx']
        del BPs['date']
        del BPs['factor']
        BPs = (BPs - BPs.min()) / (BPs.max() - BPs.min())
        BPs['target_idx'] = ti
        BPs['date'] = date
        BPs['factor'] = factor

        diff_dic = {}
        model_list = os.listdir('../../model/day2_matrix')
        code_list = []
        for i in model_list:
            code_list.append(re.findall('(.{1,}).h5', i)[0])
        # 获取账户的资金情况
        cash = context.account().cash
        valid_cash = cash.valid_cash[0]
        df_day = get_reg_kdata(reg_idx=context.reg_kdata[0], df=True, length=7)
        df_day['gains'] = df_day['close'] - df_day['open']
        df_day['volatility'] = (df_day['high'] - df_day['low'])
        df_day['ER'] = df_day['amount'] / df_day['volume']
        ti = df_day['target_idx']
        t = df_day['time']
        oi = df_day['open_interest']
        del df_day['target_idx']
        del df_day['time']
        del df_day['open_interest']
        df_day = (df_day - df_day.min()) / (df_day.max() - df_day.min())
        df_day['target_idx'] = ti
        df_day['time'] = t
        df_day['open_interest'] = oi
        for ind in range(len(model_list)):
            print(ind)
            PE = list(PEs[PEs['target_idx'] == ind].value)
            PB = list(PBs[PBs['target_idx'] == ind].value)
            # PCF = YZ[(YZ['factor'] == 'PCF') & (YZ['target_idx'] == ind)]['value']
            PS = list(PSs[PSs['target_idx'] == ind].value)
            MKV = list(MKVs[MKVs['target_idx'] == ind].value)
            # ARBR = YZ[(YZ['factor'] == 'ARBR') & (YZ['target_idx'] == ind)]['value']
            BP = list(BPs[BPs['target_idx'] == ind].value)
            open = list(df_day[df_day['target_idx'] == ind].open)
            high = list(df_day[df_day['target_idx'] == ind].high)
            low = list(df_day[df_day['target_idx'] == ind].low)
            close = list(df_day[df_day['target_idx'] == ind].close)
            volume = list(df_day[df_day['target_idx'] == ind].volume)
            amount = list(df_day[df_day['target_idx'] == ind].amount)
            # gains = list(df_day.gains)[ind]
            volatility = list(df_day[df_day['target_idx'] == ind].volatility)
            ER = list(df_day[df_day['target_idx'] == ind].ER)
            data_use = np.array([open, high, low, close, volume, amount, PE, PB, PS, MKV, BP, volatility, ER]).T
            data_use = np.array(data_use)
            # print(data_use.shape)
            data_use = data_use.reshape(data_use.shape[0], 1, data_use.shape[1])
            model = load_model('../../model/day2_matrix/' + model_list[ind])
            # input = [list(df_day.close)[ind]]
            # input = np.array(input)
            # input = input.reshape(input.shape[0], 1, 1)
            # diff_dic[code_list[ind]] = (model.predict(input)[0][0] - list(df_day.close)[ind])
            # print(data_use)
            diff_dic[code_list[ind]] = (model.predict(data_use)[0][0] - close[-1])
        diff_list = sorted(diff_dic.items(), key=lambda item: item[1], reverse=True)
        open_list = []
        target_list = []
        for i in diff_list[:5]:
            target_list.append(i[0])
        for j in diff_list[:5]:
            open_list.append(j[1])
        s = sum(open_list)
        buy_list = []
        for ii in open_list:
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
    model_list = os.listdir('../../model/day2_matrix')
    code_list = []
    for i in model_list:
        code_list.append(re.findall('(.{1,}).h5', i)[0])
    run_backtest(strategy_name='DL_day2_matrix5', file_path='.', target_list=code_list,
                 frequency='day', fre_num=1, begin_date='2021-01-01', end_date='2021-10-01', fq=0)
