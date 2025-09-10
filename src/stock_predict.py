import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys

import torch
from models.kronos import Kronos, KronosTokenizer, KronosPredictor
from utils.training_utils import create_trading_timestamps, get_device_name


def plot_prediction(kline_df, pred_df):
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 提取历史和预测数据
    historical_time = kline_df['timestamps']
    historical_close = kline_df['close']
    pred_time = pred_df['timestamps']
    pred_close = pred_df['close']

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制历史走势
    plt.plot(historical_time, historical_close, label='历史收盘价', color='blue', linewidth=1.5)
    
    # 绘制预测走势
    plt.plot(pred_time, pred_close, label='预测收盘价', color='red', linestyle='--', linewidth=1.5)
    
    # 添加标题和标签
    plt.title('股票价格走势预测', fontsize=16)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('收盘价', fontsize=14)
    
    # 美化图表
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 显示图表
    plt.show()


def predict_stock(load_online_model, lookback, pred_len, pred_date):
    # 1. Load Model and Tokenizer
    if load_online_model:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    else:
        tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("pretrained/Kronos-base")

    device = get_device_name()

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # 2. Load Data
    df = pd.read_csv("stock_data/000568_5.csv")
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    y_timestamp = create_pred_date(pred_date, pred_len)

    # 3. Prepare Data
    df_window = df.tail(lookback)

    x_df = df_window[['open', 'high', 'low', 'close', 'volume', 'amount']]
    # 使用小数后面两位小数
    x_df = x_df.round(2)
    x_timestamp = df_window['timestamps']
    # start_time = pd.Timestamp(pred_date)
    # y_timestamp = pd.date_range(
    #     start=start_time,
    #     periods=pred_len,
    #     freq='5min'
    # )
    # y_timestamp = pd.Series(y_timestamp)

    # 4. Make Prediction
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    # 5. Visualize Results
    pred_df['timestamps'] = pred_df.index
    pred_df.reset_index(drop=True, inplace=True)
    pred_df = pred_df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    print("Forecasted Data Head:")
    print(pred_df)
    plot_prediction(kline_df=df, pred_df=pred_df)

def create_pred_date(pred_date, pred_len):
    
    pred_date = datetime.datetime.strptime(pred_date, '%Y-%m-%d %H:%M:%S')
    start_date = pred_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = (pred_date + datetime.timedelta(days=4)).date().strftime('%Y-%m-%d')
    y_timestamp = create_trading_timestamps(start_date=start_date, end_date=end_date)
    
    y_timestamp = y_timestamp['timestamps'][:pred_len]
    return y_timestamp


if __name__ == "__main__":
    load_online_model = False
    lookback = 1000
    pred_len = 120
    pred_date = '2025-09-10 09:40:00'
    predict_stock(load_online_model, lookback, pred_len, pred_date)