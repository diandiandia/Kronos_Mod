from typing import List
import baostock as bs 
import pandas as pd

from data_fetch.stock_fetcher import StockFetcher


class BaostockFetcher(StockFetcher):
    def __init__(self):
        super().__init__()

    def login(self):
        self.logger.info("BaostockFetcher login")
        lg = bs.login()
        # 显示登录返回信息
        print("login respond error_code:"+lg.error_code)
        print("login respond  error_msg:"+lg.error_msg)

    def fetch_stock_ts_code_list(self) -> List[str]:
        rs = bs.query_hs300_stocks()
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        return result

    
    def fetch_by_ts_code_and_freq(self, ts_code: str, start_date: str, end_date: str, freq:str) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(ts_code, "date,time,code,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date, frequency=freq, adjustflag="2")
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        return result

    def logout(self):
        bs.logout()
