# *_*coding:utf-8 *_*  
from atrader import *  
import pandas as pd 
  
def init(context: Context):  
    # 设置初始资金为100万  
    set_backtest(initial_cash=1000000)  
  
  
def on_data(context: Context):  
    # 获取账户的持仓情况  
    positions = context.account(account_idx=0).positions  
    # 获取账户的资金情况  
    cash = context.account(account_idx=0).cash  
    # 买入开仓，市价委托  
    order_volume(account_idx=0, target_idx=0, volume=1, side=1, position_effect=1, order_type=2) 
    print("positions",pd.DataFrame(positions))
    print("cash",pd.DataFrame(cash))
  
if __name__ == '__main__':
    # 设置回测区间为2018-01-01至2018-06-30
    # 设置刷新频率为15min
    # 设置策略需要的标的为螺纹钢主力连续合约
    run_backtest(strategy_name='example_test', file_path='.', target_list=['SHFE.RB0000'], frequency='day', fre_num=1, begin_date='2018-01-01', end_date='2018-06-30')  
