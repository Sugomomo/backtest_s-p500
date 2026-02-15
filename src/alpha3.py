from utils import Alpha 
import pandas as pd 
import numpy as np 


class Alpha3(Alpha):
    def __init__(self, insts, dfs, start, end):
        super().__init__(insts, dfs, start, end) #access parent class

    def pre_compute(self,trade_range):
        for inst in self.insts:
            inst_df = self.dfs[inst]
            fast = np.where(inst_df.close.rolling(10).mean() > inst_df.close.rolling(50).mean(), 1, 0) #pandas series of 1 if true else 0
            medium = np.where(inst_df.close.rolling(20).mean() > inst_df.close.rolling(100).mean(), 1, 0) #pandas series of 1 if true else 0
            slow = np.where(inst_df.close.rolling(50).mean() > inst_df.close.rolling(200).mean(), 1, 0) #pandas series of 1 if true else 0
            alpha = fast + medium + slow
            self.dfs[inst]['alpha'] = alpha
        return 

    def post_compute(self,trade_range):
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible']\
            & (~pd.isna(self.dfs[inst]['alpha'])) 
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = self.dfs[inst].at[date, 'alpha'] #alpha_score can be pos or neg
        return forecasts, np.sum(np.abs(list(forecasts.values()))) #forecast_chips