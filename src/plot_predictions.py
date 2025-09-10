import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_prediction_files(folder_path):
    """加载预测结果文件夹中的所有CSV文件"""
    data_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_prediction.csv'):
            file_path = os.path.join(folder_path, filename)
            stock_code = filename.split('_')[0]  # 提取股票代码
            
            try:
                df = pd.read_csv(file_path)
                # 转换时间戳
                df['timestamps'] = pd.to_datetime(df['timestamps'])
                df.set_index('timestamps', inplace=True)
                data_dict[stock_code] = df
                print(f"成功加载 {filename}")
            except Exception as e:
                print(f"加载 {filename} 失败: {e}")
    
    return data_dict

def plot_stock_data(df, stock_code, save_path=None):
    """绘制股票数据的K线图和成交量图"""
    # 使用 constrained_layout 替代 tight_layout
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # K线图
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f'{stock_code} 股票预测数据', fontsize=16, fontweight='bold')
    
    # 绘制K线
    for i in range(len(df)):
        date = df.index[i]
        open_price = df['open'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        close_price = df['close'].iloc[i]
        
        # 绘制影线
        ax1.plot([date, date], [low_price, high_price], 'k-', linewidth=0.6)
        
        # 绘制实体
        if close_price >= open_price:
            color = 'red'  # 阳线
            ax1.bar(date, close_price - open_price, bottom=open_price, 
                   width=0.8, color=color, alpha=0.7)
        else:
            color = 'green'  # 阴线
            ax1.bar(date, open_price - close_price, bottom=close_price, 
                   width=0.8, color=color, alpha=0.7)
    
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['K线'], loc='upper left')
    
    # 成交量图
    ax2 = plt.subplot(gs[1], sharex=ax1)
    colors = ['red' if close >= open else 'green' 
              for close, open in zip(df['close'], df['open'])]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 成交额图
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.bar(df.index, df['amount'], color='blue', alpha=0.7, width=0.8)
    ax3.set_ylabel('成交额', fontsize=12)
    ax3.set_xlabel('时间', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 格式化x轴
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 移除 tight_layout() 调用，因为使用了 constrained_layout
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def plot_multiple_stocks(data_dict, stock_codes=None, max_stocks=6):
    """绘制多只股票的收盘价对比图"""
    if stock_codes is None:
        stock_codes = list(data_dict.keys())[:max_stocks]
    
    plt.figure(figsize=(15, 8))
    
    for stock_code in stock_codes:
        if stock_code in data_dict:
            df = data_dict[stock_code]
            plt.plot(df.index, df['close'], label=f'{stock_code}', linewidth=2)
    
    plt.title('多只股票收盘价对比', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('收盘价', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 格式化x轴
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    # 设置文件夹路径
    folder_path = 'prediction_results'
    
    # 加载数据
    print("正在加载预测数据...")
    data_dict = load_prediction_files(folder_path)
    
    if not data_dict:
        print("未找到任何预测数据文件！")
        return
    
    print(f"\n成功加载 {len(data_dict)} 只股票的数据")
    print("可用的股票代码:", list(data_dict.keys()))
    
    # 绘制单只股票的详细图表
    # print("\n绘制单只股票的详细图表...")
    # sample_stock = list(data_dict.keys())[0]  # 选择第一只股票作为示例
    # print(f"正在绘制 {sample_stock} 的图表...")
    # plot_stock_data(data_dict[sample_stock], sample_stock)
    
    # 绘制多只股票对比图
    print("\n绘制多只股票对比图...")
    plot_multiple_stocks(data_dict)
    
    # 交互式选择股票
    while True:
        user_input = input("\n请输入要查看的股票代码（输入 'q' 退出）: ").strip()
        if user_input.lower() == 'q':
            break
        
        if user_input in data_dict:
            print(f"正在绘制 {user_input} 的图表...")
            plot_stock_data(data_dict[user_input], user_input)
        else:
            print(f"未找到股票代码 {user_input}")
            print("可用的股票代码:", list(data_dict.keys()))

if __name__ == "__main__":
    main()
