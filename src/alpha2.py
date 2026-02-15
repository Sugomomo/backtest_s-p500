from utils import Alpha 
import pandas as pd 
import numpy as np 


class Alpha2(Alpha):
    def __init__(self, insts, dfs, start, end):
        super().__init__(insts, dfs, start, end) #access parent class

    def pre_compute(self,trade_range):
        self.alphas = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            alpha = -1 * (1-(inst_df.open/inst_df.close)).rolling(12).mean()
            self.alphas[inst] = alpha
        return 

    def post_compute(self,trade_range):
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.alphas[inst]
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible']\
            & (~pd.isna(self.dfs[inst]['alpha'])) 
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = self.dfs[inst].at[date, 'alpha'] #alpha_score can be pos or neg
        return forecasts, np.sum(np.abs(list(forecasts.values()))) #forecast_chips