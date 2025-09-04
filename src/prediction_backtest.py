import pandas as pd
import glob
import os

def simple_t1_analysis(prediction_dir="prediction_results"):
    """
    简单分析T+1上涨股票
    """
    # 获取所有预测结果文件
    pred_files = glob.glob(os.path.join(prediction_dir, "*_prediction.csv"))
    
    t1_up_stocks = []
    
    for file in pred_files:
        # 读取预测数据
        df = pd.read_csv(file)
        
        # 获取股票名称
        stock_name = os.path.basename(file).replace('_prediction.csv', '')
        
        # 转换时间戳
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 获取第一天和最后一天的收盘价
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        
        # 计算涨跌幅
        change_percent = ((last_close - first_close) / first_close) * 100
        
        # 判断是否T+1上涨
        is_t1_up = last_close > first_close
        
        if is_t1_up:
            t1_up_stocks.append({
                'stock': stock_name,
                'change_percent': change_percent,
                'first_close': first_close,
                'last_close': last_close
            })
        
        print(f"{stock_name}: 首日{first_close:.4f} -> 末日{last_close:.4f}, 涨幅{change_percent:.2f}%, T+1上涨: {is_t1_up}")
    
    # 按涨幅排序
    t1_up_stocks.sort(key=lambda x: x['change_percent'], reverse=True)
    
    print(f"\n=== T+1上涨股票列表 ===")
    for i, stock in enumerate(t1_up_stocks, 1):
        print(f"{i}. {stock['stock']}: 涨幅 {stock['change_percent']:.2f}%")
    
    return t1_up_stocks

def detailed_t1_analysis(prediction_dir="prediction_results", stock_data_dir="stock_data"):
    """
    详细分析T+1上涨股票，包含原始数据对比
    """
    # 获取预测文件
    pred_files = glob.glob(os.path.join(prediction_dir, "*_prediction.csv"))
    
    results = []
    
    for file in pred_files:
        stock_name = os.path.basename(file).replace('_prediction.csv', '')
        
        try:
            # 读取预测数据
            pred_df = pd.read_csv(file)
            pred_df['timestamps'] = pd.to_datetime(pred_df['timestamps'])
            
            # 读取原始数据
            original_file = os.path.join(stock_data_dir, f"{stock_name}.csv")
            if os.path.exists(original_file):
                original_df = pd.read_csv(original_file)
                original_df['timestamps'] = pd.to_datetime(original_df['timestamps'])
                original_last_close = original_df['close'].iloc[-1]
            else:
                original_last_close = None
            
            # 计算关键数据
            pred_first_close = pred_df['close'].iloc[0]
            pred_last_close = pred_df['close'].iloc[-1]
            
            # 计算各种涨跌幅
            pred_change_percent = ((pred_last_close - pred_first_close) / pred_first_close) * 100
            
            if original_last_close is not None:
                overnight_change_percent = ((pred_first_close - original_last_close) / original_last_close) * 100
                total_change_percent = ((pred_last_close - original_last_close) / original_last_close) * 100
            else:
                overnight_change_percent = None
                total_change_percent = None
            
            # 判断T+1上涨
            is_t1_up = pred_last_close > pred_first_close
            
            result = {
                'stock': stock_name,
                'original_last_close': original_last_close,
                'pred_first_close': pred_first_close,
                'pred_last_close': pred_last_close,
                'overnight_change_percent': overnight_change_percent,
                'pred_change_percent': pred_change_percent,
                'total_change_percent': total_change_percent,
                'is_t1_up': is_t1_up
            }
            
            results.append(result)
            
            print(f"{stock_name}:")
            if original_last_close is not None:
                print(f"  原始最后收盘价: {original_last_close:.4f}")
                print(f"  隔夜变化: {overnight_change_percent:.2f}%")
            print(f"  预测期间: {pred_first_close:.4f} -> {pred_last_close:.4f}")
            print(f"  预测涨幅: {pred_change_percent:.2f}%")
            print(f"  T+1上涨: {is_t1_up}")
            print()
            
        except Exception as e:
            print(f"分析 {stock_name} 时出错: {e}")
            continue
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    t1_up_stocks = results_df[results_df['is_t1_up'] == True].sort_values('pred_change_percent', ascending=False)
    
    print(f"=== T+1上涨股票排名 ===")
    for i, (_, row) in enumerate(t1_up_stocks.iterrows(), 1):
        print(f"{i}. {row['stock']}: 预测涨幅 {row['pred_change_percent']:.2f}%")
    
    return results_df

def batch_analyze_and_save(prediction_dir="prediction_results", output_file="t1_analysis_results.csv"):
    """
    批量分析并保存结果
    """
    results = detailed_t1_analysis(prediction_dir)
    
    if results is not None:
        # 保存结果
        results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n分析结果已保存到: {output_file}")
        
        # 生成统计信息
        total_stocks = len(results)
        t1_up_count = len(results[results['is_t1_up'] == True])
        t1_up_rate = (t1_up_count / total_stocks) * 100
        
        print(f"\n=== 统计信息 ===")
        print(f"总股票数: {total_stocks}")
        print(f"T+1上涨股票数: {t1_up_count}")
        print(f"T+1上涨比例: {t1_up_rate:.2f}%")
        
        # 显示前10名
        top_10 = results[results['is_t1_up'] == True].head(10)
        print(f"\n=== T+1上涨前10名 ===")
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i}. {row['stock']}: {row['pred_change_percent']:.2f}%")
    
    return results

# 使用示例
analysis_results = batch_analyze_and_save()