import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TodayBuyTomorrowSell:
    def __init__(self, prediction_dir="prediction_results", stock_data_dir="stock_data"):
        self.prediction_dir = prediction_dir
        self.stock_data_dir = stock_data_dir
        
    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['signal']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def analyze_prediction_patterns(self, pred_df):
        """分析预测数据的模式"""
        pred_df['timestamps'] = pd.to_datetime(pred_df['timestamps'])
        
        # 按日期分组
        pred_df['date'] = pred_df['timestamps'].dt.date
        daily_data = pred_df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }).reset_index()
        
        return daily_data
    
    def calculate_risk_metrics(self, daily_data):
        """计算风险指标"""
        if len(daily_data) < 2:
            return None
            
        # 波动率
        returns = daily_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(len(daily_data))
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 价格区间
        price_range = (daily_data['high'].max() - daily_data['low'].min()) / daily_data['close'].iloc[0]
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'price_range': price_range
        }
    
    def generate_trading_signals(self, original_df, pred_daily, stock_name):
        """生成交易信号"""
        signals = []
        
        # 获取原始数据最后几个交易日的技术指标
        if len(original_df) >= 20:
            recent_data = original_df.tail(20)
            last_close = original_df['close'].iloc[-1]
            last_rsi = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else 50
            last_macd = recent_data['macd'].iloc[-1] if 'macd' in recent_data.columns else 0
            last_macd_hist = recent_data['macd_histogram'].iloc[-1] if 'macd_histogram' in recent_data.columns else 0
            
            # 预测数据的首日数据
            pred_first_close = pred_daily['close'].iloc[0]
            pred_first_high = pred_daily['high'].iloc[0]
            pred_first_low = pred_daily['low'].iloc[0]
            
            # 隔夜跳空
            overnight_gap = ((pred_first_close - last_close) / last_close) * 100
            
            # T+1涨幅预测
            if len(pred_daily) >= 2:
                t1_change = ((pred_daily['close'].iloc[1] - pred_first_close) / pred_first_close) * 100
            else:
                t1_change = ((pred_daily['close'].iloc[-1] - pred_first_close) / pred_first_close) * 100
            
            # 生成买入信号
            buy_signal = False
            buy_reasons = []
            
            # 条件1: 隔夜跳空不大 (-2% 到 +3%)
            if -2 <= overnight_gap <= 3:
                buy_signal = True
                buy_reasons.append(f"隔夜跳空合理 ({overnight_gap:.2f}%)")
            
            # 条件2: RSI不过热 (30-70)
            if 30 <= last_rsi <= 70:
                buy_signal = True
                buy_reasons.append(f"RSI适中 ({last_rsi:.1f})")
            
            # 条件3: MACD趋势向好
            if last_macd > 0 and last_macd_hist > 0:
                buy_signal = True
                buy_reasons.append("MACD趋势向好")
            
            # 条件4: T+1有上涨空间
            if t1_change > 0.5:
                buy_signal = True
                buy_reasons.append(f"T+1预期上涨 {t1_change:.2f}%")
            
            # 生成卖出信号（基于预测数据）
            sell_signals = []
            
            for i, row in pred_daily.iterrows():
                if i == 0:  # 跳过第一天，因为这是买入日
                    continue
                    
                current_close = row['close']
                current_high = row['high']
                current_low = row['low']
                
                # 计算从买入点到当前点的涨幅
                from_buy_change = ((current_close - pred_first_close) / pred_first_close) * 100
                
                # 卖出条件
                sell_reasons = []
                should_sell = False
                
                # 条件1: 达到目标收益 (2%以上)
                if from_buy_change >= 2:
                    should_sell = True
                    sell_reasons.append(f"达到目标收益 {from_buy_change:.2f}%")
                
                # 条件2: 高点回落 (从高点回落超过1%)
                if i > 0:
                    prev_high = pred_daily.iloc[i-1]['high'] if i > 0 else current_high
                    high_pullback = ((prev_high - current_high) / prev_high) * 100
                    if high_pullback > 1:
                        should_sell = True
                        sell_reasons.append(f"高点回落 {high_pullback:.2f}%")
                
                # 条件3: 时间止损 (持仓超过2天)
                if i >= 2:
                    should_sell = True
                    sell_reasons.append("时间止损")
                
                if should_sell:
                    sell_signals.append({
                        'day': i,
                        'date': row['date'],
                        'price': current_close,
                        'change_percent': from_buy_change,
                        'reasons': sell_reasons
                    })
                    break  # 找到第一个卖出点就停止
            
            return {
                'buy_signal': buy_signal,
                'buy_reasons': buy_reasons,
                'sell_signals': sell_signals,
                'overnight_gap': overnight_gap,
                't1_change': t1_change,
                'entry_price': pred_first_close,
                'last_rsi': last_rsi,
                'last_macd': last_macd
            }
        
        return None
    
    def score_stock(self, signals, risk_metrics, pred_daily):
        """为股票打分"""
        if signals is None:
            return 0
            
        score = 0
        
        # 基础分数
        if signals['buy_signal']:
            score += 30
            
        # T+1涨幅分数
        if signals['t1_change'] > 0:
            score += min(signals['t1_change'] * 5, 30)  # 最多30分
            
        # 隔夜跳空分数
        if 0 < signals['overnight_gap'] <= 2:
            score += 15
        elif -1 <= signals['overnight_gap'] <= 0:
            score += 10
            
        # 技术指标分数
        if 40 <= signals['last_rsi'] <= 60:
            score += 10
        if signals['last_macd'] > 0:
            score += 10
            
        # 风险调整
        if risk_metrics:
            if risk_metrics['volatility'] < 0.05:  # 低波动率
                score += 5
            if risk_metrics['max_drawdown'] > -0.1:  # 小回撤
                score += 5
                
        return min(score, 100)  # 最高100分
    
    def screen_stocks(self, min_score=60):
        """筛选股票"""
        # 获取所有预测文件
        pred_files = glob.glob(os.path.join(self.prediction_dir, "*_prediction.csv"))
        
        recommendations = []
        
        for file in pred_files:
            stock_name = os.path.basename(file).replace('_prediction.csv', '')
            
            try:
                # 读取预测数据
                pred_df = pd.read_csv(file)
                pred_daily = self.analyze_prediction_patterns(pred_df)
                
                # 读取原始数据
                original_file = os.path.join(self.stock_data_dir, f"{stock_name}.csv")
                if not os.path.exists(original_file):
                    continue
                    
                original_df = pd.read_csv(original_file)
                original_df['timestamps'] = pd.to_datetime(original_df['timestamps'])
                original_df = self.calculate_technical_indicators(original_df)
                
                # 生成交易信号
                signals = self.generate_trading_signals(original_df, pred_daily, stock_name)
                if signals is None:
                    continue
                
                # 计算风险指标
                risk_metrics = self.calculate_risk_metrics(pred_daily)
                
                # 打分
                score = self.score_stock(signals, risk_metrics, pred_daily)
                
                if score >= min_score:
                    recommendation = {
                        'stock': stock_name,
                        'score': score,
                        'signals': signals,
                        'risk_metrics': risk_metrics,
                        'pred_daily': pred_daily
                    }
                    recommendations.append(recommendation)
                    
            except Exception as e:
                print(f"分析 {stock_name} 时出错: {e}")
                continue
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def generate_recommendation_report(self, recommendations, top_n=10):
        """生成推荐报告"""
        print("=== 今天买，明天择机卖出股票推荐 ===")
        print(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"共找到 {len(recommendations)} 只符合条件股票\n")
        
        for i, rec in enumerate(recommendations[:top_n], 1):
            signals = rec['signals']
            risk_metrics = rec['risk_metrics']
            
            print(f"{i}. {rec['stock']} (评分: {rec['score']:.0f}/100)")
            print(f"   买入价格: {signals['entry_price']:.4f}")
            print(f"   隔夜跳空: {signals['overnight_gap']:.2f}%")
            print(f"   T+1预期: {signals['t1_change']:.2f}%")
            print(f"   买入理由: {', '.join(signals['buy_reasons'])}")
            
            if signals['sell_signals']:
                sell_signal = signals['sell_signals'][0]
                print(f"   建议卖出: 第{sell_signal['day']}日 ({sell_signal['date']})")
                print(f"   预期收益: {sell_signal['change_percent']:.2f}%")
                print(f"   卖出理由: {', '.join(sell_signal['reasons'])}")
            
            if risk_metrics:
                print(f"   风险指标: 波动率{risk_metrics['volatility']:.3f}, 最大回撤{risk_metrics['max_drawdown']:.2%}")
            
            print()
        
        # 保存详细结果
        self.save_detailed_results(recommendations)
    
    def save_detailed_results(self, recommendations, filename="today_buy_tomorrow_sell_results.csv"):
        """保存详细结果"""
        detailed_results = []
        
        for rec in recommendations:
            signals = rec['signals']
            risk_metrics = rec['risk_metrics']
            
            result = {
                'stock': rec['stock'],
                'score': rec['score'],
                'entry_price': signals['entry_price'],
                'overnight_gap': signals['overnight_gap'],
                't1_change': signals['t1_change'],
                'buy_reasons': '|'.join(signals['buy_reasons']),
                'last_rsi': signals['last_rsi'],
                'last_macd': signals['last_macd'],
                'volatility': risk_metrics['volatility'] if risk_metrics else None,
                'max_drawdown': risk_metrics['max_drawdown'] if risk_metrics else None
            }
            
            if signals['sell_signals']:
                sell_signal = signals['sell_signals'][0]
                result.update({
                    'sell_day': sell_signal['day'],
                    'sell_date': sell_signal['date'],
                    'sell_price': sell_signal['price'],
                    'expected_return': sell_signal['change_percent'],
                    'sell_reasons': '|'.join(sell_signal['reasons'])
                })
            
            detailed_results.append(result)
        
        df = pd.DataFrame(detailed_results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"详细结果已保存到: {filename}")

# 使用示例
if __name__ == "__main__":
    # 创建筛选器实例
    screener = TodayBuyTomorrowSell()
    
    # 筛选股票 (最低分数60)
    recommendations = screener.screen_stocks(min_score=60)
    
    # 生成推荐报告
    screener.generate_recommendation_report(recommendations, top_n=15)
    
    print("\n=== 策略说明 ===")
    print("1. 买入条件:")
    print("   - 隔夜跳空在合理范围 (-2% 到 +3%)")
    print("   - RSI指标适中 (30-70)")
    print("   - MACD趋势向好")
    print("   - T+1有上涨空间 (>0.5%)")
    print("\n2. 卖出条件:")
    print("   - 达到目标收益 (≥2%)")
    print("   - 高点回落超过1%")
    print("   - 时间止损 (持仓超过2天)")
    print("\n3. 风险控制:")
    print("   - 综合评分低于60分的股票不推荐")
    print("   - 考虑波动率和最大回撤")
    print("   - 技术指标确认趋势")