import pandas as pd
import datetime

from data_fetch.baostock_fetcher import BaostockFetcher

def main():
    # 使用baostock_fetcher
    
    fetcher = BaostockFetcher()
    # 下载hs300股票列表
    df = fetcher.fetch_stock_ts_code_list()
    print(df)
    # 下载hs300股票数据
    start_date = '2020-01-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    freq = '5'
    for ts_code in df['code']:
        df = fetcher.fetch_by_ts_code_and_freq(ts_code, start_date=start_date, end_date=end_date, freq=freq)
        print(df)
        # 处理数据
        df['time_formatted'] = df['time'].astype(str).str[8:14].apply(lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}")
        # 2. 合并日期和时间
        df['timestamps'] = df['date'] + ' ' + df['time_formatted']

        # 3. 转换为datetime类型（可选，根据需求）
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        # 保存到csv
        df.to_csv(f"./stock_data/{ts_code}_{freq}.csv", index=False)



if __name__ == "__main__":
    main()