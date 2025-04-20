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


def load_and_preprocess_data(file_path: str, start_date: str) -> pd.DataFrame:
    """
    載入並預處理交易數據
    :param file_path: CSV文件路徑
    :param start_date: 回測起始日期 (YYYY-MM-DD)
    :return: 預處理後的DataFrame
    """
    df = pd.read_csv(file_path, sep=",")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"]) - timedelta(hours=8)
    df = df[df["datetime"] >= start_date].reset_index(drop=True)
    df["Date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["Time"] = df["datetime"].dt.strftime("%H:%M:%S")
    df.drop("datetime", axis=1, inplace=True)
    return df


def adjust_prices(
    df: pd.DataFrame, multiplier: float, additional_multipliers: List[float] = None
) -> pd.DataFrame:
    """
    調整價格數據
    :param df: 原始DataFrame
    :param multiplier: 主要價格調整乘數
    :param additional_multipliers: 額外的價格調整乘數列表
    :return: 調整後的DataFrame
    """
    price_columns = ["Open", "High", "Low", "Close"]
    df[price_columns] = df[price_columns].astype(float) * multiplier
    if additional_multipliers:
        for add_mult in additional_multipliers:
            df[price_columns] *= add_mult
    df["AVG"] = df[price_columns].mean(axis=1)
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
    REF, TARGET = config["REF"], config["TARGET"]
    trading_config = Pair_Trading_Config(
        REF,
        TARGET,
        config["open_threshold"],
        config["stop_loss_threshold"],
        config["test_second"],
        config["window_size"],
    )
    current_end_datetime = None
    # 設定日誌路徑
    log_path = f"./Trading_Log_NEW/{REF}{TARGET}/_{REF}{TARGET}_{config['window_size']}length_Trading_log/"
    os.makedirs(log_path, exist_ok=True)
    print(f"日誌目錄已創建: {log_path}")

    ########
    open_close_log_path = f"./Open_Close_Log/{REF}_{TARGET}/_{REF}_{TARGET}_{config['window_size']}length_Trading_log"
    os.makedirs(open_close_log_path, exist_ok=True)
    print(f"已創建開平倉紀錄目錄 : {open_close_log_path}")

    # 初始化交易機器人
    period = "morning" if period_choice == 1 else "night"
    spreader = Spreader(
        trading_config, REF, TARGET, log_path, period, open_close_log_path
    )
    record_time = spreader.record_time
    # 載入並預處理數據
    start_date = config["backtest_start_date"]
    ref_df = load_and_preprocess_data(config["ref_file"], start_date)
    target_df = load_and_preprocess_data(config["target_file"], start_date)

    # 調整價格
    ref_df = adjust_prices(ref_df, config["ref_multiplier"])
    target_df = adjust_prices(
        target_df, config["target_multiplier"], config["target_additional_multipliers"]
    )
    print(ref_df)
    print(target_df)

    # 設定交易時間
    time_open, time_end = config["trading_hours"][period]

    # 紀錄交易結果
    # spreader.predictor.existing_df.to_csv(f"./TABLE/{REF}_{TARGET}_formation_table.csv", index=False)

    # 模擬交易
    print("simulate ")
    print(ref_df)
    simulate_trading(ref_df, target_df, spreader, REF, TARGET, time_open, time_end)

    #####
    # daily_df = pd.DataFrame(spreader.daily_data)
    # daily_df['DateTime'] = pd.to_datetime(daily_df['Date'] + ' ' + daily_df['Timestamp'])
    # visualize_spread_and_volume(daily_df)
    # print(daily_df.head(50))
    # visualize_trading_data(daily_df, "./daily_visual_chart.png")
    # 保存預測結果
    spreader.predictor.existing_df.to_csv(
        f"./TABLE/{REF}_{TARGET}_formation_table.csv", index=False
    )


def simulate_trading(
    ref_df: pd.DataFrame,
    target_df: pd.DataFrame,
    spreader: Spreader,
    REF: str,
    TARGET: str,
    time_open: str,
    time_end: str,
):
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
        ref_date, ref_time, ref_price, ref_volume = ref_df.iloc[i][
            ["Date", "Time", "Close", "TotalVolume"]
        ]
        target_date, target_time, target_price, target_volume = target_df.iloc[j][
            ["Date", "Time", "Close", "TotalVolume"]
        ]

        # 如果兩邊的日期/時間不完全匹配，則以較早的那一筆先移動
        if ref_date < target_date or (
            ref_date == target_date and ref_time < target_time
        ):
            i += 1
            continue
        elif ref_date > target_date or (
            ref_date == target_date and ref_time > target_time
        ):
            j += 1
            continue
        else:
            # 當 ref_date 與 target_date 及時間完全一致時，判斷是否屬於交易時段
            if in_trading_session(ref_date, ref_time, time_open, time_end):
                print(f"模擬交易: {ref_date} {ref_time}")
                print(f"參考價格: {ref_price}, 目標價格: {target_price}")
                # 執行交易模擬，並傳入 time_end 作為收盤時刻參數
                spreader.local_simulate(
                    ref_date, ref_time, REF, ref_price, ref_price, ref_volume, time_end
                )
                spreader.local_simulate(
                    target_date,
                    target_time,
                    TARGET,
                    target_price,
                    target_price,
                    target_volume,
                    time_end,
                )
            i += 1
            j += 1


if __name__ == "__main__":

    REF_CBOT = ["ZT", "ZF", "ZN", "ZB", "TN"]
    TARGET_CBOT = ["ZT", "ZF", "ZN", "ZB", "TN"]
    ################ CME
    REF_CME = ["SI", "NQ", "PL", "NG", "HG", "LE", "HE", "GC", "GF", "CL"]
    TARGET_CME = ["SI", "NQ", "PL", "NG", "HG", "LE", "HE", "GC", "GF", "CL"]
    REF_CME_MULT = [5000, 20, 50, 10000, 25000, 40000, 40000, 100, 50000, 1000]
    TARGET_CME_MULT = [5000, 20, 50, 10000, 25000, 40000, 40000, 100, 50000, 1000]

    CBOT_REF_MULT = [2000, 1000, 1000, 1000, 1000]
    CBOT_TARGET_MULT = [2000, 1000, 1000, 1000, 1000]

    record = [
        ("ZF", "TN"),
        ("ZF", "ZB"),
        ("ZF", "ZN"),
        ("ZN", "TN"),
        ("ZN", "ZB"),
        ("ZT", "TN"),
        ("ZT", "ZB"),
        ("ZT", "ZF"),
        ("ZT", "ZN"),
        ("ZB", "TN"),
    ]

    #######
    # CBOT_REF_MULT = [50, 2000, 50, 1000, 1000, 1000, 50, 5, 1000]
    # CBOT_TARGET_MULT = [50, 2000, 50, 1000, 1000, 1000, 50, 5, 1000]
    for i in range(len(REF_CBOT)):
        for j in range(len(TARGET_CBOT)):

            if REF_CBOT[i] == TARGET_CBOT[j]:
                continue

            flag = 0
            print(REF_CBOT[i] + " " + TARGET_CBOT[j])
            for k in range(len(record)):
                if REF_CBOT[i] == record[k][0] and TARGET_CBOT[j] == record[k][1]:
                    print("jump")
                    flag = 1
            if flag:
                continue

            REF = REF_CBOT[i]
            TARGET = TARGET_CBOT[j]
            multi_ref = CBOT_REF_MULT[i]
            multi_target = CBOT_TARGET_MULT[j]
            # multi_ref = CME_REF_MULT[i]
            # multi_target = CME_TARGET_MULT[j]

            config = {
                "REF": f"CBOT_{REF}",
                "TARGET": f"CBOT_{TARGET}",
                "open_threshold": 1.5,  # 開倉門檻
                "stop_loss_threshold": 10,  # 平倉門檻
                "test_second": 60,  # 測試秒數 (收集幾分k)
                "window_size": 150,  # kbar窗口大小
                "trading_hours": {
                    "morning": ("00:00:01", "23:59:00"),
                    "night": ("17:01:00", "15:59:00"),  # 新增：晚間交易時間
                },
                # 'ref_file': f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.{REF} HOT-Minute-Trade.txt", # 配對1的交易數據
                # 'target_file': f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.{TARGET} HOT-Minute-Trade.txt", # 配對2的交易數據
                "ref_file": f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CBOT\\CBOT.{REF} HOT-Minute-Trade.txt",
                "target_file": f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CBOT\\CBOT.{TARGET} HOT-Minute-Trade.txt",
                "ref_multiplier": multi_ref,  # 配對1價格乘數
                "target_multiplier": multi_target,  # 配對2價格乘數
                "target_additional_multipliers": [30, 2],  # 新增：配對2額外價格乘數 (美金匯率 跟 張數)
                "backtest_start_date": "2019-01-01",  # 新增：回測起始日期
                "observation_period": 150,
            }

            period_choice = 0  # 0 表示晚間交易, 1 表示早間交易
            main(config, period_choice)

    # 如果要測試不同的開倉閾值，可以使用以下代碼
    # open_threshold_list = [1.5, 2, 2.5, 3, 3.5]
    # for open_threshold in open_threshold_list:
    #     config['open_threshold'] = open_threshold
    #     main(config, period_choice)
