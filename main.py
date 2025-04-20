import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import List, Tuple
import module.Johanson_class as Jo_class
from module.spreader import Spreader
from config import Pair_Trading_Config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from datetime import time
import math
# 設定當前工作目錄，放在import其他路徑模組之前
sys.path.append("./module")
os.chdir(sys.path[0])

record_time = []



def visualize_spread_and_volume(daily_df, prev_w1=1, prev_w2=-1):
    # Separate CME_SI and CME_GC prices
    cme_si = daily_df[daily_df['Symbol'] == 'CME_SI']
    cme_gc = daily_df[daily_df['Symbol'] == 'CME_GC']

    # Align data on DateTime
    merged = pd.merge(cme_si, cme_gc, on='DateTime', suffixes=('_SI', '_GC'))

    # Calculate weights
    w1, w2 = calculate_weights(merged['Price_SI'], merged['Price_GC'], prev_w1, prev_w2)

    # Calculate spread
    merged['Spread'] = w1 * np.log(merged['Price_SI']) + w2 * np.log(merged['Price_GC'])

    # Plot spread and volume
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot spread
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Spread', color='blue')
    ax1.plot(merged['DateTime'], merged['Spread'], color='blue', label='Spread')
    ax1.tick_params(axis='y', labelcolor='blue')


    # Plot volume on a secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volume', color='orange')
    ax2.bar(merged['DateTime'], merged['Volume_SI'] + merged['Volume_GC'], color='orange', alpha=0.5, label='Total Volume')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Final adjustments
    plt.title('Spread and Volume Over Time')
    fig.tight_layout()
    plt.legend()
    plt.show()

    return w1, w2


def visualize_p_values_with_metrics(timestamps, p_values):
    print(f'record time{record_time}')
    # Combine date and time, then convert to datetime format
    datetime_series = pd.to_datetime([f"{date} {time}" for date, time in timestamps])

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({'Timestamp': datetime_series, 'P-Value': p_values})

    avg_p_value = np.mean(p_values)
    median_p_value = np.median(p_values)
    std_p_value = np.std(p_values)
    if len(p_values) != 0:
        coint_ratio = sum(p < 0.05 for p in p_values) / len(p_values)
    else:
        coint_ratio = 0


    print("Overall Cointegration Metrics:")
    print(f"Average : {avg_p_value:.4f}")
    print(f"Median : {median_p_value:.4f}")
    print(f'Std {std_p_value}')
    print(f"Cointegration Ratio (P-Value < 0.05): {coint_ratio:.2%}")


    plt.figure(figsize=(14, 7))
    plt.plot(df['Timestamp'], df['P-Value'], color='b', marker='o', label='P-Value')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')

    # Add horizontal lines for mean, median, and standard deviation
    plt.axhline(y=avg_p_value, color='g', linestyle='-', label=f'Mean: {avg_p_value:.4f}')
    # plt.axhline(y=median_p_value, color='orange', linestyle='-', label=f'Median: {median_p_value:.4f}')
    plt.axhline(y=avg_p_value + std_p_value, color='purple', linestyle='--',
                label=f'Mean + 1 Std: {avg_p_value + std_p_value:.4f}')
    plt.axhline(y=avg_p_value - std_p_value, color='purple', linestyle='--',
                label=f'Mean - 1 Std: {avg_p_value - std_p_value:.4f}')

    # Setting date format on x-axis
   # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))  # 每隔一天顯示一次

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Adjust as necessary for readability

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('P-Value')
    plt.title('P-Value Over Time for Cointegration Test')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # dates = [ts[0] for ts in timestamps]
    # times = [ts[1] for ts in timestamps]
    #
    # df = pd.DataFrame({'Date': dates, 'Timestamp': times, 'P-Value': p_values})
    #
    #
    # avg_p_value = np.mean(p_values)
    # median_p_value = np.median(p_values)
    # if len(p_values) != 0:
    #     coint_ratio = sum(p < 0.05 for p in p_values) / len(p_values)
    # else:
    #     coint_ratio = 0
    #
    #
    # print("Overall Cointegration Metrics:")
    # print(f"Average P-Value: {avg_p_value:.4f}")
    # print(f"Median P-Value: {median_p_value:.4f}")
    # print(f"Cointegration Ratio (P-Value < 0.05): {coint_ratio:.2%}")
    #
    #
    # plt.figure(figsize=(14, 7))
    # plt.scatter(df['Timestamp'], df['P-Value'], color='b', label='P-Value')
    # plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    #
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    #
    # plt.xticks(ticks=df['Timestamp'][::len(df) // 10], rotation=45, ha='right')
    #
    # plt.xlabel('Timestamp')
    # plt.ylabel('P-Value')
    # plt.title('P-Value Over Time for Cointegration Test')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

def load_and_preprocess_data(file_path: str, start_date: str) -> pd.DataFrame:
    """
    載入並預處理交易數據
    :param file_path: CSV文件路徑
    :param start_date: 回測起始日期 (YYYY-MM-DD)
    :return: 預處理後的DataFrame
    """
    df = pd.read_csv(file_path, sep=",")
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']) - timedelta(hours=8)
    df = df[df['datetime'] >= start_date].reset_index(drop=True)
    df['Date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['Time'] = df['datetime'].dt.strftime('%H:%M:%S')
    df.drop('datetime', axis=1, inplace=True)
    return df

def adjust_prices(df: pd.DataFrame, multiplier: float, additional_multipliers: List[float] = None) -> pd.DataFrame:
    """
    調整價格數據
    :param df: 原始DataFrame
    :param multiplier: 主要價格調整乘數
    :param additional_multipliers: 額外的價格調整乘數列表
    :return: 調整後的DataFrame
    """
    price_columns = ['Open', 'High', 'Low', 'Close']
    df[price_columns] = df[price_columns].astype(float) * multiplier
    if additional_multipliers:
        for add_mult in additional_multipliers:
            df[price_columns] *= add_mult
    df['AVG'] = df[price_columns].mean(axis=1)
    return df

def get_end_datetime(ref_file, target_file):
    return 0
def update_current_end_datetime():
    pass

def observation():
    pass

def get_start_trading_time():
    return 0

def main(config: dict, period_choice: int):
    # 設定交易對和配置
    REF, TARGET = config['REF'], config['TARGET']
    trading_config = Pair_Trading_Config(REF, TARGET, config['open_threshold'], 
                                         config['stop_loss_threshold'], config['test_second'],
                                         config['window_size'])
    current_end_datetime = None
    # 設定日誌路徑
    log_path = f"./Trading_Log_NEW/{REF}{TARGET}/_{REF}{TARGET}_{config['window_size']}length_Trading_log/"
    os.makedirs(log_path, exist_ok=True)
    print(f"日誌目錄已創建: {log_path}")

    ########
    open_close_log_path = f"./Open_Close_Log/{REF}_{TARGET}/_{REF}_{TARGET}_{config['window_size']}length_Trading_log"
    os.makedirs(open_close_log_path,exist_ok=True)
    print(f"已創建開平倉紀錄目錄 : {open_close_log_path}")





    # 初始化交易機器人
    period = 'morning' if period_choice == 1 else 'night'
    spreader = Spreader(trading_config, REF, TARGET, log_path, period, open_close_log_path)
    record_time = spreader.record_time
    # 載入並預處理數據
    start_date = config['backtest_start_date']
    ref_df = load_and_preprocess_data(config['ref_file'], start_date)
    target_df = load_and_preprocess_data(config['target_file'], start_date)

    # 調整價格
    ref_df = adjust_prices(ref_df, config['ref_multiplier'])
    target_df = adjust_prices(target_df, config['target_multiplier'], config['target_additional_multipliers'])
    print(ref_df)
    print(target_df)


    # 設定交易時間
    time_open, time_end = config['trading_hours'][period]





    # 紀錄交易結果
    #spreader.predictor.existing_df.to_csv(f"./TABLE/{REF}_{TARGET}_formation_table.csv", index=False)

    # 模擬交易
    print('simulate ')
    print(ref_df)
    simulate_trading(ref_df, target_df, spreader, REF, TARGET, time_open, time_end)



    #####
    # daily_df = pd.DataFrame(spreader.daily_data)
    # daily_df['DateTime'] = pd.to_datetime(daily_df['Date'] + ' ' + daily_df['Timestamp'])
    # visualize_spread_and_volume(daily_df)
    #print(daily_df.head(50))
    # visualize_trading_data(daily_df, "./daily_visual_chart.png")
    # 保存預測結果
    spreader.predictor.existing_df.to_csv(f"./TABLE/{REF}_{TARGET}_formation_table.csv", index=False)


# def simulate_trading(ref_df: pd.DataFrame,
#                      target_df: pd.DataFrame,
#                      spreader: Spreader,
#                      REF: str,
#                      TARGET: str,
#                      time_open: str,
#                      time_end: str):
#     """
#     模擬交易過程，支援「跨日」(例如從 17:01 到隔日 15:59)。
#     """
#     def combine_datetime(date_str, time_str):
#         """結合日期、時間成 datetime"""
#         return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
#
#     def get_trading_range(date_str, time_open, time_end):
#         """
#         回傳給定 date_str 當天的交易區間 (start_datetime, end_datetime)。
#         如果 time_open > time_end，表示跨日，需要把 end_datetime 加一天。
#         """
#         start_dt = combine_datetime(date_str, time_open)
#         end_dt = combine_datetime(date_str, time_end)
#         if time_open > time_end:
#             end_dt += timedelta(days=1)
#         return start_dt, end_dt
#
#     i, j = 0, 0
#
#     # 當兩個DataFrame中的資料都還沒跑完時
#     while i < len(ref_df) and j < len(target_df):
#         # 取出 ref_df 第 i 筆資料
#         ref_row = ref_df.iloc[i]
#         ref_date = ref_row['Date']
#         ref_time = ref_row['Time']
#         ref_price = ref_row['Close']
#         ref_volume = ref_row['TotalVolume']
#         ref_datetime = combine_datetime(ref_date, ref_time)
#
#         # 取出 target_df 第 j 筆資料
#         tgt_row = target_df.iloc[j]
#         tgt_date = tgt_row['Date']
#         tgt_time = tgt_row['Time']
#         tgt_price = tgt_row['Close']
#         tgt_volume = tgt_row['TotalVolume']
#         tgt_datetime = combine_datetime(tgt_date, tgt_time)
#
#         # 若兩筆時間不一致則依較早者往前移動
#         if ref_datetime < tgt_datetime:
#             i += 1
#             continue
#         elif ref_datetime > tgt_datetime:
#             j += 1
#             continue
#         else:
#             # 當前資料的時間相同，開始模擬交易
#
#             # 如果交易時間跨日（例如夜盤：time_open > time_end）
#             # 將屬於隔日早盤（時間 < time_end）的數據，歸屬到前一日的 session
#             if time_open > time_end:
#                 # 注意：這裡比較字串也可，因 "09:00:00" < "15:59:00"
#                 if ref_time < time_end:
#                     # 調整 session 日期為前一天
#                     session_date = (datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
#                 else:
#                     session_date = ref_date
#             else:
#                 session_date = ref_date
#
#             # 以調整後的 session_date 取得交易區間
#             start_dt, end_dt = get_trading_range(session_date, time_open, time_end)
#
#             # 若當前時間在交易區間內才模擬交易
#             if start_dt <= ref_datetime <= end_dt:
#                 print(f"\n模擬交易: {ref_datetime}")
#                 print(f"  參考價格: {ref_price}, 目標價格: {tgt_price}")
#                 spreader.local_simulate(ref_date, ref_time, REF, ref_price, ref_price, ref_volume, time_end)
#                 spreader.local_simulate(tgt_date, tgt_time, TARGET, tgt_price, tgt_price, tgt_volume, time_end)
#
#             i += 1
#             j += 1
def simulate_trading(ref_df: pd.DataFrame, target_df: pd.DataFrame, spreader: Spreader,
                     REF: str, TARGET: str, time_open: str, time_end: str):
    """
    模擬交易過程，支援跨日交易（例如從 17:01 到隔日 15:59）。

    參數:
      - ref_df: 參考資產的 DataFrame（包含 'Date', 'Time', 'Close', 'TotalVolume'）
      - target_df: 目標資產的 DataFrame（同上）
      - spreader: 交易機器人
      - time_open: 交易開始時間字串 (例如 "17:01:00")
      - time_end: 交易結束時間字串 (例如 "15:59:00")
    """

    def in_trading_session(date_str, time_str, time_open, time_end):
        """
        判斷當前時間是否屬於交易區間。
        如果 time_open < time_end，則區間在同一日；若 time_open > time_end，表示跨日，
        則只要時間大於等於 time_open 或小於等於 time_end 都算在交易區間內。
        """
        t = datetime.strptime(time_str, "%H:%M:%S").time()
        open_time = datetime.strptime(time_open, "%H:%M:%S").time()
        end_time = datetime.strptime(time_end, "%H:%M:%S").time()
        if open_time < end_time:
            return open_time < t <= end_time
        else:
            # 跨日情況：只要 t >= open_time 或 t <= end_time
            return t >= open_time or t <= end_time

    i, j = 0, 0
    while i < len(ref_df) and j < len(target_df):
        # 取得參考與目標資料各自的 Date, Time, Close, TotalVolume
        ref_date, ref_time, ref_price, ref_volume = ref_df.iloc[i][['Date', 'Time', 'Close', 'TotalVolume']]
        target_date, target_time, target_price, target_volume = target_df.iloc[j][
            ['Date', 'Time', 'Close', 'TotalVolume']]

        # 如果兩邊的日期/時間不完全匹配，則以較早的那一筆先移動
        if ref_date < target_date or (ref_date == target_date and ref_time < target_time):
            i += 1
            continue
        elif ref_date > target_date or (ref_date == target_date and ref_time > target_time):
            j += 1
            continue
        else:
            # 當 ref_date 與 target_date 及時間完全一致時，判斷是否屬於交易時段
            if in_trading_session(ref_date, ref_time, time_open, time_end):
                print(f"模擬交易: {ref_date} {ref_time}")
                print(f"參考價格: {ref_price}, 目標價格: {target_price}")
                # 執行交易模擬，並傳入 time_end 作為收盤時刻參數
                spreader.local_simulate(ref_date, ref_time, REF, ref_price, ref_price, ref_volume, time_end)
                spreader.local_simulate(target_date, target_time, TARGET, target_price, target_price, target_volume,
                                        time_end)
            i += 1
            j += 1


# def simulate_trading(ref_df: pd.DataFrame, target_df: pd.DataFrame, spreader: Spreader,
#                      REF: str, TARGET: str, time_open: str, time_end: str, ):
#     """
#     模擬交易過程
#     :param ref_df: 參考資產DataFrame
#     :param target_df: 目標資產DataFrame
#     :param spreader: Spreader實例
#     :param REF: 參考資產代碼
#     :param TARGET: 目標資產代碼
#     :param time_open: 交易開始時間
#     :param time_end: 交易結束時間
#     """
#
#
#
#
# ### original
#     i, j = 0, 0
#     while i < len(ref_df) and j < len(target_df):
#         ref_date, ref_time, ref_price, ref_volume = ref_df.iloc[i][['Date', 'Time', 'Close', 'TotalVolume']]
#         target_date, target_time, target_price, target_volume = target_df.iloc[j][['Date', 'Time', 'Close', 'TotalVolume']]
#
#         if ref_date < target_date or (ref_date == target_date and ref_time < target_time):
#             i += 1
#         elif ref_date > target_date or (ref_date == target_date and ref_time > target_time):
#             j += 1
#         else:
#             if time_open < ref_time <= time_end:
#                 print(f"模擬交易: {ref_date} {ref_time}")
#                 print(f"參考價格: {ref_price}, 目標價格: {target_price}")
#                 spreader.local_simulate(ref_date, ref_time, REF, ref_price, ref_price, ref_volume, time_end)
#                 spreader.local_simulate(target_date, target_time, TARGET, target_price, target_price, target_volume, time_end)
#             i += 1
#             j += 1
#     def convert_to_datetime(date_str, time_str):
#         return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
#
#     i, j = 0, 0
#     while i < len(ref_df) and j < len(target_df):
#         ref_date, ref_time, ref_price, ref_volume = ref_df.iloc[i][['Date', 'Time', 'Close', 'TotalVolume']]
#         target_date, target_time, target_price, target_volume = target_df.iloc[j][
#             ['Date', 'Time', 'Close', 'TotalVolume']]
#
#         # ref_datetime = convert_to_datetime(ref_date, ref_time)
#         # target_datetime = convert_to_datetime(target_date, target_time)
#
#         # 計算跨日的交易時間範圍
#         start_datetime = convert_to_datetime(ref_date, time_open)
#         if time_open > time_end:  # 跨日的情況
#             end_datetime = convert_to_datetime(ref_date, time_end) + timedelta(days=1)
#             print('start datetime')
#             print(f"start {start_datetime} end {end_datetime}")
#         else:
#             end_datetime = convert_to_datetime(ref_date, time_end)
#
#         # 如果時間在範圍之外，調整索引
#         if ref_datetime < target_datetime:
#             i += 1
#         elif ref_datetime > target_datetime:
#             j += 1
#         else:
#             if start_datetime <= ref_datetime <= end_datetime:
#                 print(f"模擬交易: {ref_date} {ref_time}")
#                 print(f"參考價格: {ref_price}, 目標價格: {target_price}")
#                 spreader.local_simulate(ref_date, ref_time, REF, ref_price, ref_price, ref_volume, time_end)
#                 spreader.local_simulate(target_date, target_time, TARGET, target_price, target_price, target_volume,
#                                         time_end)
#             i += 1
#             j += 1
#
#     print("start")
#     i, j = 0, 0
#     while i+1 < len(ref_df) and j+1 < len(target_df):
#         ref_date, ref_time, ref_price, ref_volume = ref_df.iloc[i][['Date', 'Time', 'Close', 'TotalVolume']]
#         ref_next_date, ref_next_time,
#         print('ref date')
#         print(ref_date)
#         print('ref time')
#         print(ref_time)
#         print('ref price')
#         print(ref_price)
#         print('ref volume')
#         print(ref_volume)
#
#         target_date, target_time, target_price, target_volume = target_df.iloc[j][['Date', 'Time', 'Close', 'TotalVolume']]
#
#         if ref_date < target_date or (ref_date == target_date and ref_time < target_time):
#             i += 1
#         elif ref_date > target_date or (ref_date == target_date and ref_time > target_time):
#             j += 1
#         else:
#             if time_open < ref_time <= time_end:
#                 print(f"模擬交易: {ref_date} {ref_time}")
#                 print(f"參考價格: {ref_price}, 目標價格: {target_price}")
#                 spreader.local_simulate(ref_date, ref_time, REF, ref_price, ref_price, ref_volume, time_end)
#                 spreader.local_simulate(target_date, target_time, TARGET, target_price, target_price, target_volume, time_end)
#             i += 1
#             j += 1

if __name__ == "__main__":
    # REF_CME = ['CME_HG', 'CME_GF', 'CME_LE', 'CME_NQ', 'CME_SI']
    # TARGET_CME = ['CME_GC', 'CME_LE', 'CME_HE', 'CME_GC', 'CME_PL']
    # REF = ['HG', 'GF', 'LE', 'NQ', 'SI']
    # TARGET = ['GC', 'LE', 'HE', 'GC', 'PL']




    ################# CBOT
    # REF_CBOT = ['ZW','ZT','ZS','ZN','ZF','ZB','ZC','YM','TN']
    # TARGET_CBOT = ['ZW','ZT','ZS','ZN','ZF','ZB','ZC','YM','TN']

    REF_CBOT = ['ZT','ZF','ZN','ZB','TN']
    TARGET_CBOT = ['ZT','ZF','ZN','ZB','TN']
    ################ CME
    REF_CME = ['SI','NQ','PL','NG','HG','LE','HE','GC','GF','CL']
    TARGET_CME = ['SI','NQ','PL','NG','HG','LE','HE','GC','GF','CL']
    REF_CME_MULT = [5000, 20, 50, 10000, 25000, 40000, 40000, 100, 50000, 1000]
    TARGET_CME_MULT = [5000, 20, 50, 10000, 25000, 40000, 40000, 100, 50000, 1000]

    CBOT_REF_MULT = [2000,1000,1000,1000,1000]
    CBOT_TARGET_MULT = [2000,1000,1000,1000,1000]

    record = [('ZF','TN'),('ZF','ZB'),('ZF','ZN'),('ZN','TN'),('ZN','ZB'),('ZT','TN'),('ZT','ZB'),('ZT','ZF'),('ZT','ZN'),('ZB','TN')]

    #######
    # CBOT_REF_MULT = [50, 2000, 50, 1000, 1000, 1000, 50, 5, 1000]
    # CBOT_TARGET_MULT = [50, 2000, 50, 1000, 1000, 1000, 50, 5, 1000]
    for i in range(len(REF_CBOT)):
        for j in range(len(TARGET_CBOT)):


            if REF_CBOT[i] == TARGET_CBOT[j]:
                continue

            flag = 0
            print(REF_CBOT[i] + ' ' + TARGET_CBOT[j])
            for k in range(len(record)):
                if REF_CBOT[i] == record[k][0] and TARGET_CBOT[j] == record[k][1]:
                    print('jump')
                    flag = 1
            if flag:
                continue

            REF = REF_CBOT[i]
            TARGET = TARGET_CBOT[j]
            multi_ref = CBOT_REF_MULT[i]
            multi_target = CBOT_TARGET_MULT[j]

            config = {
                 'REF': f'CBOT_{REF}',
                 'TARGET': f'CBOT_{TARGET}',
                'open_threshold': 1.5, #開倉門檻
                'stop_loss_threshold': 10, #平倉門檻
                'test_second': 60, #測試秒數 (收集幾分k)
                'window_size': 150, #kbar窗口大小
                'trading_hours': {
                    'morning': ('00:00:01','23:59:00'),
                    'night': ('17:01:00', '15:59:00') # 新增：晚間交易時間
                },
               # 'ref_file': f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.{REF} HOT-Minute-Trade.txt", # 配對1的交易數據
               # 'target_file': f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.{TARGET} HOT-Minute-Trade.txt", # 配對2的交易數據
                'ref_file':f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CBOT\\CBOT.{REF} HOT-Minute-Trade.txt",
                'target_file':f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CBOT\\CBOT.{TARGET} HOT-Minute-Trade.txt",
              #  'ref_file':r"C:\Users\Henry\Downloads\Touchance\Touchance\Taiwan_Futures\TWF.NVF HOT-Minute-Trade.txt",
                #'target_file':r"C:\Users\Henry\Downloads\Touchance\Touchance\Taiwan_Futures\TWF.NUF HOT-Minute-Trade.txt",
                #'ref_file':"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PairTradingOriginal\\test_trade\\SI_FROM_20201101_010000_TO_20201130_170000.txt",
                #'target_file':"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PairTradingOriginal\\test_trade\\GC_FROM_20201101_010000_TO_20201130_170000.txt",
                #'ref_file': "C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\SI_FROM_20140131_010000_TO_20140131_170000.txt",
                #'target_file': "C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\GC_FROM_20140131_010000_TO_20140131_170000.txt",
                'ref_multiplier': multi_ref , # 配對1價格乘數
                'target_multiplier': multi_target, # 配對2價格乘數
                'target_additional_multipliers': [30, 2],  # 新增：配對2額外價格乘數 (美金匯率 跟 張數)
                'backtest_start_date': '2019-01-01',# 新增：回測起始日期
                'observation_period': 150
            }

            period_choice = 0  # 0 表示晚間交易, 1 表示早間交易
            main(config, period_choice)

    #
    # visualize_p_values_with_metrics(Jo_class.timestep, Jo_class.p_value_list)
    # print(Jo_class.timestep)
    # print(Jo_class.p_value_list)



    # 如果要測試不同的開倉閾值，可以使用以下代碼
    # open_threshold_list = [1.5, 2, 2.5, 3, 3.5]
    # for open_threshold in open_threshold_list:
    #     config['open_threshold'] = open_threshold
    #     main(config, period_choice)