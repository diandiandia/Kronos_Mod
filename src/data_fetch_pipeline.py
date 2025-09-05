import pandas as pd
import datetime

from data_fetch.baostock_fetcher import BaostockFetcher
from data_fetch.tushare_fetcher import TushareFetcher

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
        # 处理数据time列20220729093500000为2022-07-29 09:35:00格式
        # 将20220729转换成2022-07-29
        df['date'] = df['time'].astype(str).str[:4] + '-' + df['time'].astype(str).str[4:6] + '-' + df['time'].astype(str).str[6:8]
        # 将093500转换成09:35:00
        # Correct time slicing to get HH:MM:SS format by limiting seconds to 2 digits
        df['time_new'] = df['time'].astype(str).str[8:10] + ':' + df['time'].astype(str).str[10:12] + ':' + df['time'].astype(str).str[12:14]
        df['timestamps'] = df['date'] + ' ' + df['time_new']
        # 3. 转换为datetime类型（可选，根据需求）
        # Specify timestamp format to avoid parsing warnings
        df['timestamps'] = pd.to_datetime(df['timestamps'], format='%Y-%m-%d %H:%M:%S')
        df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        # 保存到csv
        df.to_csv(f"./stock_data/{ts_code}_{freq}.csv", index=False)



if __name__ == "__main__":
    main()