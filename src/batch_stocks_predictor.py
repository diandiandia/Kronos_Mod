import os
import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import torch
from models.kronos import Kronos, KronosTokenizer, KronosPredictor
from utils.training_utils import create_trading_timestamps, get_device_name


def plot_prediction(kline_df, pred_df, stock_name):
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
    plt.title(f'{stock_name} 股票价格走势预测', fontsize=16)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('收盘价', fontsize=14)
    
    # 美化图表
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 显示图表
    plt.show()


def create_pred_date(pred_date, pred_len):
    
    pred_date = datetime.datetime.strptime(pred_date, '%Y-%m-%d %H:%M:%S')
    start_date = pred_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = (pred_date + datetime.timedelta(days=4)).date().strftime('%Y-%m-%d')
    y_timestamp = create_trading_timestamps(start_date=start_date, end_date=end_date)
    
    y_timestamp = y_timestamp['timestamps'][:pred_len]
    return y_timestamp


def predict_single_stock(predictor, stock_file, lookback, pred_len, pred_date, save_results=True):
    """
    对单个股票进行预测
    
    Args:
        predictor: KronosPredictor实例
        stock_file: 股票数据文件路径
        lookback: 历史数据长度
        pred_len: 预测长度
        pred_date: 预测开始日期
        save_results: 是否保存结果
    
    Returns:
        tuple: (股票名称, 预测结果DataFrame)
    """
    # 获取股票名称
    stock_name = Path(stock_file).stem
    
    try:
        # 2. 加载数据
        df = pd.read_csv(stock_file)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 检查数据是否足够
        if len(df) < lookback:
            print(f"警告: {stock_name} 数据不足 (需要{lookback}条，实际{len(df)}条)，跳过预测")
            return stock_name, None
        
        y_timestamp = create_pred_date(pred_date, pred_len)
        
        # 3. 准备数据
        df_window = df.tail(lookback)
        
        x_df = df_window[['open', 'high', 'low', 'close', 'volume', 'amount']]
        # 使用小数后面两位小数
        x_df = x_df.round(2)
        x_timestamp = df_window['timestamps']
        
        # 4. 进行预测
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False  # 批量预测时不显示详细进度
        )
        
        # 5. 处理预测结果
        pred_df['timestamps'] = pred_df.index
        pred_df.reset_index(drop=True, inplace=True)
        pred_df = pred_df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        print(f"{stock_name} 预测完成")
        print(f"预测数据前5行:")
        print(pred_df.head())
        
        # 6. 可视化结果
        # plot_prediction(kline_df=df, pred_df=pred_df, stock_name=stock_name)
        
        # 7. 保存结果
        if save_results:
            # 创建结果目录
            results_dir = "prediction_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存预测结果
            result_file = os.path.join(results_dir, f"{stock_name}_prediction.csv")
            pred_df.to_csv(result_file, index=False)
            print(f"预测结果已保存到: {result_file}")
        
        return stock_name, pred_df
        
    except Exception as e:
        print(f"预测 {stock_name} 时发生错误: {str(e)}")
        return stock_name, None


def batch_predict_stocks(stock_data_dir="stock_data", load_online_model=False, lookback=1000, 
                        pred_len=120, pred_date='2025-09-04 09:35:00', save_results=True):
    """
    批量预测多个股票
    
    Args:
        stock_data_dir: 股票数据目录
        load_online_model: 是否加载在线模型
        lookback: 历史数据长度
        pred_len: 预测长度
        pred_date: 预测开始日期
        save_results: 是否保存结果
    
    Returns:
        dict: 股票名称到预测结果的映射
    """
    # 1. 加载模型和分词器
    if load_online_model:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    else:
        tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("pretrained/Kronos-base")

    device = get_device_name()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=128)
    
    # 2. 获取所有股票数据文件
    stock_files = glob.glob(os.path.join(stock_data_dir, "*.csv"))
    stock_files.sort()  # 按文件名排序
    
    print(f"找到 {len(stock_files)} 个股票数据文件")
    
    # 3. 批量预测
    results = {}
    successful_predictions = 0
    
    for i, stock_file in enumerate(stock_files):
        print(f"\n[{i+1}/{len(stock_files)}] 正在预测: {Path(stock_file).name}")
        
        stock_name, pred_df = predict_single_stock(
            predictor=predictor,
            stock_file=stock_file,
            lookback=lookback,
            pred_len=pred_len,
            pred_date=pred_date,
            save_results=save_results
        )
        
        if pred_df is not None:
            results[stock_name] = pred_df
            successful_predictions += 1
        
        # 添加适当的延迟，避免内存问题
        if i % 10 == 0 and i > 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n批量预测完成!")
    print(f"成功预测: {successful_predictions}/{len(stock_files)} 个股票")
    
    return results


if __name__ == "__main__":
    # 配置参数
    load_online_model = False
    lookback = 1000
    pred_len = 120
    pred_date = datetime.datetime.today().strftime('%Y-%m-%d') + ' 09:35:00'
    # pred_date = '2025-09-10 09:35:00'
    
    # 执行批量预测
    results = batch_predict_stocks(
        stock_data_dir="stock_data",
        load_online_model=load_online_model,
        lookback=lookback,
        pred_len=pred_len,
        pred_date=pred_date,
        save_results=True
    )
    
    # 打印总结
    print("\n=== 批量预测总结 ===")
    print(f"总股票数: {len(results)}")
    print("预测的股票:")
    for stock_name in results.keys():
        print(f"  - {stock_name}")