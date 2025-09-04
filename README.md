# 基于Kronos的股票T+1股票推荐

## 预测

### 拉取历史数据

先拉取昨天之前的`freq=5min`的数据，使用如下脚本：

```python
python src/data_fetch_pipeline.py
```

脚本会在`stock_data`文件夹下生成`交易所.股票代码_数据间隔时间.csv`文件。

### 预测

脚本可以使用在线模型与本地模型，通过load_online_model配置在线模型后者本地模型：

```python
    if load_online_model:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    else:
        tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("pretrained/Kronos-base")
```

Kronos提供了small，base模型，更具实际电脑算例，配置模型。

使用如下脚本进行单个股票预测

```python
python src/stock_predict.py
```

注意替换股票数据

```python
df = pd.read_csv("stock_data\sh.600426_5.csv")
```

使用如下脚本批量预测股票

```python
python src/batch_stocks_predictor.py
```

预测结束，会生成`prediction_results`文件夹，里面包含股票的预测数据。

### 获取推荐信息

获取到预测数据后，可以使用如下脚本对股票信息进行汇总，获取T+1（今天买入，明天择机卖出）推荐股票信息。

```python
python .\src\prediction_backtest.py
```

根据输出信息，获取推荐股票信息：

```log
=== 今天买，明天择机卖出股票推荐 ===
筛选时间: 2025-09-04 14:23:54
共找到 3 只符合条件股票

1. sh.600025_5 (评分: 66/100)
   买入价格: 9.1922
   隔夜跳空: 0.35%
   T+1预期: 0.18%
   买入理由: 隔夜跳空合理 (0.35%), RSI适中 (55.6)
   建议卖出: 第2日 (2025-09-06)
   预期收益: 0.43%
   卖出理由: 时间止损
   风险指标: 波动率0.001, 最大回撤0.00%

2. sh.600000_5 (评分: 65/100)
   买入价格: 13.9945
   隔夜跳空: 1.56%
   T+1预期: -0.51%
   买入理由: 隔夜跳空合理 (1.56%), RSI适中 (63.6), MACD趋势向好
   卖出理由: 时间止损
   风险指标: 波动率0.006, 最大回撤-0.03%

3. sh.600027_5 (评分: 64/100)
   买入价格: 5.2864
   隔夜跳空: -1.74%
   T+1预期: 0.87%
   买入理由: 隔夜跳空合理 (-1.74%), RSI适中 (58.3), MACD趋势向好, T+1预期上涨 0.87%
   建议卖出: 第2日 (2025-09-06)
   预期收益: 1.67%
   卖出理由: 时间止损
   风险指标: 波动率0.001, 最大回撤0.00%

详细结果已保存到: today_buy_tomorrow_sell_results.csv

=== 策略说明 ===
1. 买入条件:
   - 隔夜跳空在合理范围 (-2% 到 +3%)
   - RSI指标适中 (30-70)
   - MACD趋势向好
   - T+1有上涨空间 (>0.5%)

2. 卖出条件:
   - 达到目标收益 (≥2%)
   - 高点回落超过1%
   - 时间止损 (持仓超过2天)

3. 风险控制:
   - 综合评分低于60分的股票不推荐
   - 考虑波动率和最大回撤
   - 技术指标确认趋势
```



## 训练
