from utils import Alpha
import pandas as pd
import numpy as np 

class Alpha1(Alpha):
    def __init__(self, insts, dfs, start, end):
        super().__init__(insts, dfs, start, end) #access parent class

    def pre_compute(self,trade_range):
        self.op4s= {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            # Alpha1 raw feature: signed intraday pressure scaled by volume/range.
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3 
            self.op4s[inst] = op4 
        return 
    
    def post_compute(self,trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]['op4'] = self.op4s[inst] #alignment
            temp.append(self.dfs[inst]['op4'])

        # Cross-sectional normalize op4 each day, then smooth to create alpha signal.
        temp_df = pd.concat(temp, axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace(np.inf, np.nan).replace(-np.inf, np.nan) #replace inf to 0
        zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x) #zscore function
        cszcre_df = temp_df.ffill().apply(zscore, axis =1,raw=True) 

        alphas = []
        for inst in self.insts:
            self.dfs[inst]['alpha'] = cszcre_df[inst].rolling(12).mean() * -1
            alphas.append(self.dfs[inst]['alpha'])
        alphadf = pd.concat(alphas, axis =1)
        alphadf.columns = self.insts
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        # Mask out non-eligible instruments so ranking only reflects tradable names each day.
        masked_df = self.alphadf/self.eligiblesdf 
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)#set non trading to nan
        num_eligibles = self.eligiblesdf.sum(axis=1) #num eligibles in universe
        # Daily cross-sectional rank, then map bottom/top quartiles to -1/+1 forecasts.
        rankdf = masked_df.rank(axis=1, method="average", na_option="keep",ascending=True)
        shortdf = rankdf.apply(lambda col: col <= num_eligibles.values/4 ,axis =0,raw=True) #if 10, short 1,2 
        longdf = rankdf.apply(lambda col: col > np.ceil(num_eligibles-num_eligibles/4),axis =0,raw=True)#if 10, long 9, 10
        forecast_df = -1*shortdf.astype(np.int32) + longdf.astype(np.int32)
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date): #this child will run instead of parent class 
        # Return a full cross-section vector aligned to self.insts for vectorized engine.
        forecasts = self.forecast_df.loc[date].values 
        return forecasts
