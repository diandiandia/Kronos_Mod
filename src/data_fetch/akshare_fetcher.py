import akshare as ak
import pandas as pd

from typing import List
from data_fetch.stock_fetcher import StockFetcher


class AkshareFetcher(StockFetcher):
    def __init__(self):
        super().__init__()

    def login(self):
        self.logger.info("Login to Akshare")

    def fetch_stock_ts_code_list(self) -> List[str]:
        try:
            df = ak.index_stock_cons_csindex(symbol="000300")
            columns = {
                '成分券代码': 'code',
                '成分券名称': 'name',
                '日期': 'date',
            }
            df.rename(columns=columns, inplace=True)
            df = df[['code', 'name', 'date']]
            self.logger.info(f"Successfully fetched stock ts code list from Akshare")
            return df
        except:
            self.logger.error(f"Failed to fetch stock ts code list from Akshare")
            return []

    
    def fetch_by_ts_code_and_freq(self, ts_code: str, start_date: str, end_date: str, freq:str) -> pd.DataFrame:
        try:
            start_date = start_date + " 09:00:00"
            end_date = end_date + " 19:00:00"
            df = ak.stock_zh_a_hist_min_em(symbol=ts_code, start_date=start_date, end_date=end_date, period=freq, adjust='qfq')
            # 返回：时间，开盘，收盘，最高，最低，涨跌幅，涨跌额，成交量，成交额，振幅，换手率，需要替换成英文名称
            columns = {
                '时间': 'time',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '涨跌幅': 'change',
                '涨跌额': 'change_amount',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '换手率': 'turnover',
            }
            df.rename(columns=columns, inplace=True)
            # 修改df['time'] 2025-07-29 09:35:00为20250729093500
            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y%m%d%H%M%S')
            df['volume'] = df['volume'] * 100
            df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            self.logger.info(f"Successfully fetched data for {ts_code} from Akshare")
            return df
        except:
            self.logger.error(f"Failed to fetch data for {ts_code} from Akshare")
            return pd.DataFrame()

    
    def logout(self):
        self.logger.info("Logout from Akshare")