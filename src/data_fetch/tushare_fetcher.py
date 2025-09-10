from datetime import datetime, timedelta
from typing import List
import tushare as ts
import pandas as pd

from data_fetch.stock_fetcher import StockFetcher


class TushareFetcher(StockFetcher):
    def __init__(self):
        super().__init__()

    def login(self, token="c477c6691a86fa6f410f520f8f2e59f195ba9cb93b76384047de3d8d"):
        self.logger.info("BaostockFetcher login")
        ts.set_token(token)
        self.pro = ts.pro_api()

    def fetch_stock_ts_code_list(self) -> List[str]:
        try:
            
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            df = self.pro.index_weight(
                index_code='000300.SH',
                start_date=start_date,
                end_date=end_date,
            )
            self.logger.info("Successfully retrieved HS300 stocks from tushare.")
            # 修改con_code为code
            df.rename(columns={'con_code': 'code'}, inplace=True)
            df = df.drop_duplicates(subset=['code'])
            return df[["code", "trade_date", "weight"]]
        except Exception as e:
            self.logger.error(f"Error retrieving HS300 stocks: {e}")
            return None

    
    def fetch_by_ts_code_and_freq(self, ts_code: str, start_date: str, end_date: str, freq:str) -> pd.DataFrame:
        try:
            start_date = start_date + " 09:00:00"
            end_date = end_date + " 19:00:00"

            df = self.pro.stk_mins(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                freq='5min',
            )
            # trade_time为2023-08-25 15:00:00转换为time列20230825150000
            df['time'] = df['trade_time'].astype(str).str.replace('-', '').astype(str).str.replace(' ', '').astype(str).str.replace(':', '')
            # 修改vol为volume
            df.rename(columns={'vol': 'volume'}, inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving stock {ts_code} data: {e}")
            return None

    def logout(self):
        self.pro.logout()
