

from abc import abstractmethod
import logging
from typing import List

import pandas as pd


class StockFetcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.login()


    @abstractmethod
    def login(self):
        pass

    @abstractmethod
    def fetch_stock_ts_code_list(self) -> List[str]:
        pass

    
    @abstractmethod
    def fetch_by_ts_code_and_freq(self, ts_code: str, start_date: str, end_date: str, freq:str) -> pd.DataFrame:
        pass

    @abstractmethod
    def logout(self):
        pass

    def __del__(self):
        self.logout()
