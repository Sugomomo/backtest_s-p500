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
            # Alpha2: mean-reversion style signal from open-vs-close relationship.
            alpha = -1 * (1-(inst_df.open/inst_df.close)).rolling(12).mean()
            self.alphas[inst] = alpha
        return 

    def post_compute(self,trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.alphas[inst]
            temp.append(self.dfs[inst]["alpha"])
        alphadf = pd.concat(temp, axis=1)
        alphadf.columns = self.insts
        alphadf = alphadf.ffill()
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        return 

    def compute_signal_distribution(self, eligibles, date):
        # Return full cross-section vector aligned to self.insts.
        forecasts = self.alphadf.loc[date].values
        return forecasts
