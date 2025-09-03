import pandas as pd
import matplotlib.pyplot as plt
import sys

import torch
from models.kronos import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    # 设置中文字体
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
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


# 1. Load Model and Tokenizer
online = False
if online:
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
else:
    tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("pretrained/Kronos-base")

if torch.cuda.is_available():
    device_name = "cuda:0"
elif torch.mps.is_available():
    device_name = "mps"
elif torch.xpu.is_available():
    device_name = "xpu:0"
else:
    device_name = "cpu"

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device=device_name, max_context=512)

# 3. Prepare Data
df = pd.read_csv("stock_data/sh.600000_5.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

# x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
# x_timestamp = df.loc[:lookback-1, 'timestamps']
# y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

df_window = df.tail(lookback)

df_window = df[-lookback-48: -48]

xx_df = df_window[['open', 'high', 'low', 'close', 'volume', 'amount']]
xx_timestamp = df_window['timestamps']
start_time = pd.Timestamp('2025-09-03 09:35:00')
yy_timestamp = pd.date_range(
    start=start_time,
    periods=pred_len,
    freq='5min'
)
yy_timestamp = pd.Series(yy_timestamp)  

# 4. Make Prediction
pred_df = predictor.predict(
    df=xx_df,
    x_timestamp=xx_timestamp,
    y_timestamp=yy_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
pred_df['timestamps'] = pred_df.index
pred_df.reset_index(drop=True, inplace=True)
print("Forecasted Data Head:")
print(df_window)
print(pred_df)

# Combine historical and forecasted data for plotting
kline_df = df[-lookback:]


# visualize
plot_prediction(kline_df, pred_df)