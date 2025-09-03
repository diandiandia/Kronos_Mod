import pandas as pd
import matplotlib.pyplot as plt
import torch
from model.kronos import Kronos, KronosTokenizer, KronosPredictor

online = True

def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
print(f"Online is {online}")
if online:
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
else:
    tokenizer = KronosTokenizer.from_pretrained("../pretrained/kronos-Tokenizer-base")
    model = Kronos.from_pretrained("../pretrained/Kronos-base")

# 2. Instantiate Predictor
if torch.cuda.is_available():
    device_name = "cuda:0"
elif torch.mps.is_available():
    device_name = "mps"
elif torch.xpu.is_available():
    device_name = "xpu:0"
else:
    device_name = "cpu"

print("Using device:", device_name)


predictor = KronosPredictor(model, tokenizer, device=device_name, max_context=64)

# 3. Prepare Data
# df = pd.read_csv("./data/XSHG_5min_600977.csv")
df = pd.read_csv("D:/Projects/Kronos/stock_data/sh.600000_5.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])
# 数据使用小数点后2位
df['open'] = df['open'].round(2)
df['high'] = df['high'].round(2)
df['low'] = df['low'].round(2)
df['close'] = df['close'].round(2)
df['volume'] = df['volume'].round(2)
df['amount'] = df['amount'].round(2)

# lookback = 400
# pred_len = 120

# x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
# x_timestamp = df.loc[:lookback-1, 'timestamps']
# y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

lookback = 60
pred_len = 60

x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']]  # 使用最新的历史数据



x_timestamp = df.loc[-lookback:, 'timestamps']  # 使用最新的时间戳

# 生成2025-09-03的交易时间戳（9:35-15:00，每5分钟一个数据点）
start_time = pd.Timestamp('2025-09-03 09:35:00')
y_timestamp = pd.date_range(
    start=start_time,
    periods=pred_len,
    freq='5min'
)
y_timestamp = pd.Series(y_timestamp)  # 转换为Series类型

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
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

