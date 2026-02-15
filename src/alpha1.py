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

        temp_df = pd.concat(temp, axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace(np.inf, 0).replace(-np.inf, 0) #replace inf to 0
        zscore = lambda x: (x - np.mean(x))/np.std(x) #zscore function
        cszcre_df = temp_df.ffill().apply(zscore, axis =1) 
        for inst in self.insts:
            self.dfs[inst]['alpha'] = cszcre_df[inst].rolling(12).mean() * -1
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible'] & \
                (~pd.isna(self.dfs[inst]['alpha'])) #alpha non-null
        return 

    def compute_signal_distribution(self, eligibles, date): #this child will run instead of parent class 
        alpha_scores = {} #forecast for every inst in our universe
        for inst in eligibles:
            alpha_scores[inst] = self.dfs[inst].at[date, 'alpha'] 
        alpha_scores = {k:v for k,v in sorted(alpha_scores.items(), key=lambda pair: pair[1])}
        alpha_long = list(alpha_scores.keys())[-int(len(eligibles)/4):]
        alpha_short = list(alpha_scores.keys())[:int(len(eligibles)/4)]
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0) #1(long), -1(short), 0(no trade)
        return forecasts, np.sum(np.abs(list(forecasts.values()))) #forecast_chips