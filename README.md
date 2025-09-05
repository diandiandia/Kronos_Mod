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

## 获取数据

从如下地址下载免费数据：

```shell
wget https://github.com/chenditc/investment_data/releases/download/2023-10-08/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C qlib_data/cn_data --strip-components=1
```

查看数据qlib_data/cn_data/calendars/day.txt中的数据起始信息：

```shell
2005-01-04
2005-01-05
……
2025-09-03
2025-09-04
```

修改config.py中的信息：

```python
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = "qlib_data/cn_data"  # 你下载的数据保存的地址
        self.instrument = 'csi300' # 查看cn_data/instruments/*文件，查看有哪些，csi300，csi500或者其他，按照这个信息改。
        
        
        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2005-01-04"   # 数据起始地址
        self.dataset_end_time = '2025-09-04'     # 数据结束地址
        
        
        self.train_time_range = ["2005-01-04", "2024-12-31"]  # 扩展训练集到2005年开始
        self.val_time_range = ["2024-09-01", "2025-06-30"]    # 调整验证集
        self.test_time_range = ["2025-04-01", "2025-09-04"]   # 更新测试集到最新数据
        self.backtest_time_range = ["2025-07-01", "2025-09-04"] # 更新回测时间范围
        
        # 预训练的模型地址
        self.pretrained_tokenizer_path = "pretrained/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "pretrained/Kronos-small"
```

### 处理数据

```python
python .\src\finetune\qlib_data_preprocess.py
```

处理完后会生成pkl文件。

### 训练

token单卡训练：

```python
python src/train_tokenizer_single.py
```

训练信息：

```shell
使用设备: cpu
Loading weights from local directory
模型参数量: 4.0M
开始训练...
批次大小: 50
有效总批次大小: 50
创建数据加载器...
[TRAIN] Pre-computing sample indices...
[TRAIN] Found 1303568 possible samples. Using 100000 per epoch.
[VAL] Pre-computing sample indices...
[VAL] Found 27629 possible samples. Using 20000 per epoch.
训练数据集大小: 100000, 验证数据集大小: 20000
数据加载器创建完成。训练步数/epoch: 2000, 验证步数: 400
[Epoch 1/30, Step 100/2000] LR 0.000021, Loss: -0.0223
```

模型训练：

```python
python src/train_predictor_single.py
```

训练信息：

```shell
 epoch.
[VAL] Pre-computing sample indices...
[VAL] Found 27629 possible samples. Using 20000 
```
