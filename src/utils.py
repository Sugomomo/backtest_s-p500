import lzma 
import dill as pickle 
import pandas as pd
import numpy as np
from datetime import timedelta
from copy import deepcopy
from collections import defaultdict
import time
from functools import wraps

def timeme(func):
    @wraps(func)
    def timediff(*args,**kwargs):
        a = time.time()
        result = func(*args,**kwargs)
        b = time.time()
        print(f'@timeme: {func.__name__} took {b-a} seconds')
        return result
    return timediff

def load_pickle(path):
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj): #save data as python.obj
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def get_pnl_stats(date, prev, portfolio_df, insts, idx,dfs):
    day_pnl = 0 
    nominal_ret = 0
    for inst in insts:
        units = portfolio_df.at[idx-1, "{} units".format(inst)] #prev day size
        if units != 0:
            delta = dfs[inst].at[date, "close"] - dfs[inst].at[prev, "close"] #change in price
            inst_pnl = delta * units #change in price is negative, and if short pnl will be positive
            day_pnl += inst_pnl
            nominal_ret += portfolio_df.at[idx-1, "{} w".format(inst)] * dfs[inst].at[date, "ret"] #nominal portfolio ret: portfolio allocation  * ret of insts (return on nominal exposure)
    capital_ret = nominal_ret * portfolio_df.at[idx - 1, "leverage"] #entire capital return is nominal_ret * prev day leverage (return on portfolio)
    portfolio_df.at[idx, 'capital'] = portfolio_df.at[idx -1, 'capital'] + day_pnl 
    portfolio_df.at[idx, 'day_pnl'] = day_pnl
    portfolio_df.at[idx, 'nominal_ret'] = nominal_ret
    portfolio_df.at[idx, 'capital_ret'] = capital_ret
    return day_pnl, capital_ret
class AbstractImplementationException(Exception):
    pass

class Alpha():

    def __init__(self, insts, dfs, start, end, portfolio_vol = 0.20):
        self.insts = insts
        self.dfs = deepcopy(dfs) #ensure every alpha get their own same copy of dfs
        self.start = start #backtest period
        self.end = end 
        self.portfolio_vol = portfolio_vol #target vol

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range)\
        .reset_index().rename(columns={'index':'datetime'})
        
        weight_cols = [f"{inst} w" for inst in self.insts]
        unit_cols = [f"{inst} units" for inst in self.insts]
        base_cols = ['capital', 'day_pnl', 'capital_ret', 'nominal_ret', 'nominal', 'leverage']
        all_new_cols = base_cols + weight_cols + unit_cols
        zeros_df = pd.DataFrame(0.0, index=portfolio_df.index, columns=all_new_cols)
        portfolio_df = pd.concat([portfolio_df, zeros_df], axis=1)

        portfolio_df.at[0, 'capital'] = 10000.0
        return portfolio_df
    
    def pre_compute(self,trade_range):
        pass

    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException('No concrete implementation for signal generation')
        #each respective alphas will have their own, hence this main class will not be implemented
    
    def compute_meta_info(self, trade_range):
        self.pre_compute(trade_range=trade_range)
        def is_any_one(x):
            return int(np.any(x))
        
        closes, eligibles,vols, rets =[], [],[], []
        for inst in self.insts:
            df = pd.DataFrame(index = trade_range)
            inst_vol = (-1 + self.dfs[inst]['close']/self.dfs[inst]['close'].shift(1)).rolling(30).std() #simple vol using return, vol must be calculated before ffill as there will be duplicate in ffill
            #match indices to fill data (aval date <-> dfs[inst] of date)
            self.dfs[inst] = df.join(self.dfs[inst]).ffill()#fill down 
            self.dfs[inst]['ret'] = -1 + self.dfs[inst]['close']/self.dfs[inst]['close'].shift(1) #return
            self.dfs[inst]['vol'] = inst_vol
            self.dfs[inst]['vol'] = self.dfs[inst]['vol'].ffill().fillna(0) #fill most recent vol for missing
            self.dfs[inst]['vol'] = np.where(self.dfs[inst]['vol'] < 0.005, 0.005, self.dfs[inst]['vol']) #threshold
            sampled = self.dfs[inst]['close'] != self.dfs[inst]['close'].shift(1).bfill() #check tdy vs ytd
            eligible = sampled.rolling(5).apply(is_any_one,raw=True).fillna(0) #only eligible for trade by grouping 5 sample store T(1)/F(0) into new pd.series (1/0) 
            eligibles.append(eligible.astype(int) & (self.dfs[inst]['close'] > 0).astype(int)) #other condition with eligible
            closes.append(self.dfs[inst]['close'])
            vols.append(self.dfs[inst]['vol'])
            rets.append(self.dfs[inst]['ret'])

        self.eligiblesdf = pd.concat(eligibles,axis=1) 
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes,axis=1) 
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols,axis=1) 
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets,axis=1) 
        self.retdf.columns = self.insts


        self.post_compute(trade_range=trade_range)
        return 
    
    def get_strat_scaler(self,target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253) #take the last annualized variance
        return target_vol / ann_realized_vol * ewstrats[-1] #use the strategy scaler already implementing to find ann_realized_vol
        
    @timeme
    def run_simulation(self):
        print('running backtest')
        start = self.start + timedelta(hours=5)
        end = self.end + timedelta(hours=5)
        date_range = pd.date_range(start,end,freq="D") #trading date_range, not every ticker exist at pt of time
        self.compute_meta_info(trade_range=date_range)
        portfolio_df = self.init_portfolio_settings(trade_range = date_range)
        # ewmas: EWMA of strategy variance proxy from capital_ret^2 (not per-instrument vol)
        # ewstrats: EWMA of applied strategy scaling factor (strat_scalar), used for smoothing leverage scaling
        self.ewmas , self. ewstrats = [0.01], [1] #default
        self.strat_scalars = [] #target_vol / realized_vol as scalar for position 
        for i in portfolio_df.index:
            date = portfolio_df.at[i, "datetime"]
            eligibles = [inst for inst in self.insts if self.dfs[inst].at[date,'eligible']] #trading universe
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]
            strat_scalar = 2 #default for first day 

            if i !=0: #if not first day
                date_prev = portfolio_df.at[i-1, "datetime"]
                strat_scalar = self.get_strat_scaler(
                    target_vol = self.portfolio_vol,
                    ewmas=self.ewmas,
                    ewstrats =self.ewstrats #previous strat-scaler
                )

                day_pnl, capital_ret = get_pnl_stats(date=date, prev =date_prev, 
                portfolio_df=portfolio_df, insts =self.insts, idx=i,dfs=self.dfs)

                self.ewmas.append(0.06 * (capital_ret**2) + 0.94 * self.ewmas[-1] if capital_ret !=0 else self.ewmas[-1]) # update EWMA variance estimate using squared portfolio return
                self.ewstrats.append(0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if capital_ret !=0 else self.ewstrats[-1]) # update EWMA of scaling factor (not return volatility)

            self.strat_scalars.append(strat_scalar)
            forecasts, forecast_chips = self.compute_signal_distribution(eligibles, date)

            for inst in non_eligibles: #assign non_eligibles weight and unit(allocation) to 0
                portfolio_df.at[i, "{} w".format(inst)] = 0
                portfolio_df.at[i, "{} units".format(inst)] = 0
            vol_target = (self.portfolio_vol / np.sqrt(253)) * portfolio_df.at[i,'capital'] #annual_vol(in dollar) = daily_vol * capital

            nominal_tot = 0
            for inst in eligibles:
                forecast = forecasts[inst]
                scaled_forecast = forecast / forecast_chips if forecast_chips != 0 else 0

                position = strat_scalar * scaled_forecast * vol_target / \
                    (self.dfs[inst].at[date,'vol'] * self.dfs[inst].at[date,'close']) 
                # position units = target dollar risk / instrument dollar volatility (signed by forecast)
                #instrumental level vol targeting 
                '''
                vol * close say how much dollar PnL one unit typically moves.
                scaled_forecast * vol_target is how much dollar risk you want from that instrument.
                position = target risk / risk per unit gives how many units to hold
                '''


                portfolio_df.at[i, inst + " units"] = position #new col for position size 
                nominal_tot += abs(position * self.dfs[inst].at[date,'close']) #absolute sizing of position in terms of dollar 

            for inst in eligibles:
                units = portfolio_df.at[i, inst + " units"]
                nominal_inst = units * self.dfs[inst].at[date,'close'] #nominal_val of insts (not abs because can be negative for short)
                if nominal_tot > 0:
                    inst_w = nominal_inst / nominal_tot
                else:
                    inst_w = 0.0
                portfolio_df.at[i, inst + " w"] = inst_w

            portfolio_df.at[i, "nominal"] = nominal_tot #total gross notional exposure.
            portfolio_df.at[i , "leverage"] = nominal_tot / portfolio_df.at[i, "capital"] #total leverage
        return portfolio_df.set_index('datetime', drop=True)


class Portfolio(Alpha):

    def __init__(self, insts, dfs, start, end ,stratdfs):
        super().__init__(insts, dfs, start,end) 
        self.stratdfs = stratdfs
    
    def post_compute(self, trade_range): 
        self.positions = {}
        for inst in self.insts:
            inst_weight = pd.DataFrame(index=trade_range)
            for i in range(len(self.stratdfs)): 
                inst_weight[i] = self.stratdfs[i]['{} w'.format(inst)] \
                * self.stratdfs[i]['leverage']#portfolio weight for every sub strategy (position sizing)
                inst_weight[i] = inst_weight[i].ffill().fillna(0.0)
            self.positions[inst] = inst_weight

    def compute_signal_distribution(self, eligibles, date):
        forecasts = defaultdict(float)
        for inst in self.insts:
            for i in range(len(self.stratdfs)):
                forecasts[inst] += self.positions[inst].at[date,i] * (1/len(self.stratdfs)) #parity risk allocation
        return forecasts, np.sum(np.abs(list(forecasts.values())))


def _get_pnl_stats(last_weights, last_units, prev_close, portfolio_i, ret_row, portfolio_df):
                day_pnl = np.sum(last_units * prev_close * ret_row) #dollar pnl 
                nominal_ret = np.dot(last_weights * ret_row) 
                capital_ret = nominal_ret * portfolio_df.at[portfolio_i - 1, "leverage"] #entire capital return is nominal_ret * prev day leverage (return on portfolio)
                portfolio_df.at[portfolio_i, 'capital'] = portfolio_df.at[portfolio_i -1, 'capital'] + day_pnl 
                portfolio_df.at[portfolio_i, 'day_pnl'] = day_pnl
                portfolio_df.at[portfolio_i, 'nominal_ret'] = nominal_ret
                portfolio_df.at[portfolio_i, 'capital_ret'] = capital_ret
                return day_pnl, capital_ret

class EfficientAlpha():
    def __init__(self, insts, dfs, start, end, portfolio_vol = 0.20):
        self.insts = insts
        self.dfs = deepcopy(dfs) #ensure every alpha get their own same copy of dfs
        self.start = start #backtest period
        self.end = end 
        self.portfolio_vol = portfolio_vol #target vol

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range)\
        .reset_index().rename(columns={'index':'datetime'})
        
        weight_cols = [f"{inst} w" for inst in self.insts]
        unit_cols = [f"{inst} units" for inst in self.insts]
        base_cols = ['capital', 'day_pnl', 'capital_ret', 'nominal_ret', 'nominal', 'leverage']
        all_new_cols = base_cols + weight_cols + unit_cols
        zeros_df = pd.DataFrame(0.0, index=portfolio_df.index, columns=all_new_cols)
        portfolio_df = pd.concat([portfolio_df, zeros_df], axis=1)

        portfolio_df.at[0, 'capital'] = 10000.0
        return portfolio_df
    
    def pre_compute(self,trade_range):
        pass

    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException('No concrete implementation for signal generation')
        #each respective alphas will have their own, hence this main class will not be implemented
    
    def get_strat_scaler(self,target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253) #take the last annualized variance
        return target_vol / ann_realized_vol * ewstrats[-1] #use the strategy scaler already implementing to find ann_realized_vol
    
    def compute_meta_info(self, trade_range):
        self.pre_compute(trade_range=trade_range)
        def is_any_one(x):
            return int(np.any(x))
        
        closes, eligibles,vols, rets =[], [],[], []
        for inst in self.insts:
            df = pd.DataFrame(index = trade_range)
            inst_vol = (-1 + self.dfs[inst]['close']/self.dfs[inst]['close'].shift(1)).rolling(30).std() #simple vol using return, vol must be calculated before ffill as there will be duplicate in ffill
            #match indices to fill data (aval date <-> dfs[inst] of date)
            self.dfs[inst] = df.join(self.dfs[inst]).ffill()#fill down 
            self.dfs[inst]['ret'] = -1 + self.dfs[inst]['close']/self.dfs[inst]['close'].shift(1) #return
            self.dfs[inst]['vol'] = inst_vol
            self.dfs[inst]['vol'] = self.dfs[inst]['vol'].ffill().fillna(0) #fill most recent vol for missing
            self.dfs[inst]['vol'] = np.where(self.dfs[inst]['vol'] < 0.005, 0.005, self.dfs[inst]['vol']) #threshold
            sampled = self.dfs[inst]['close'] != self.dfs[inst]['close'].shift(1).bfill() #check tdy vs ytd
            eligible = sampled.rolling(5).apply(is_any_one,raw=True).fillna(0) #only eligible for trade by grouping 5 sample store T(1)/F(0) into new pd.series (1/0) 
            eligibles.append(eligible.astype(int) & (self.dfs[inst]['close'] > 0).astype(int)) #other condition with eligible
            closes.append(self.dfs[inst]['close'])
            vols.append(self.dfs[inst]['vol'])
            rets.append(self.dfs[inst]['ret'])

        self.eligiblesdf = pd.concat(eligibles,axis=1) 
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes,axis=1) 
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols,axis=1) 
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets,axis=1) 
        self.retdf.columns = self.insts

        self.post_compute(trade_range=trade_range)
        return

    @timeme
    def run_simulation(self):
        self.portfolio_df = self.init_portfolio_settings()
        start = self.start + timedelta(hours=5)
        end = self.end + timedelta(hours=5)
        date_range = pd.date_range(start,end,freq="D") #trading date_range, not every ticker exist at pt of time
        self.compute_meta_info(trade_range=date_range)
        self.portfolio_df = self.init_portfolio_settings(trade_range = date_range)

        units_held, weights_held = [], []
        close_prev = None 
        self.ewmas , self. ewstrats = [0.01], [1] #default
        self.strat_scalars = [] #target_vol / realized_vol as scalar for position 
        portfolio_df = self.portfolio_df

        for data in self.zip_data_generator():
            portfolio_i = data["portfolio_i"] 
            portfolio_row = data["portfolio_row"]
            ret_i = data["ret_i"] 
            ret_row = data["ret_row"]
            close_row = data["close_row"]
            eligibles_row = data["eligibles_row"]
            vol_row = data["vol_row"]
            strat_scalar = 2

            if portfolio_i != 0:
                strat_scalar = self.get_strat_scaler(
                    target_vol = self.portfolio_vol,
                    ewmas=self.ewmas,
                    ewstrats =self.ewstrats #previous strat-scaler
                )
                day_pnl, capital_ret = _get_pnl_stats(last_weights=weights_held[-1], last_units=units_held[-1], prev_close=close_prev, 
                                                     portfolio_i=portfolio_i, ret_row=ret_row, portfolio_df= self.portfolio_df)

                self.ewmas.append(0.06 * (capital_ret**2) + 0.94 * self.ewmas[-1] if capital_ret !=0 else self.ewmas[-1]) # update EWMA variance estimate using squared portfolio return
                self.ewstrats.append(0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if capital_ret !=0 else self.ewstrats[-1]) # update EWMA of scaling factor (not return volatility)

            self.strat_scalars.append(strat_scalar)
            forecasts, forecasts_chips = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            vol_target = (self.portfolio_vol / np.sqrt(253)) * portfolio_df.at[portfolio_i,'capital'] #annual_vol(in dollar) = daily_vol * capital
            positions = strat_scalar * forecasts / forecasts_chips * vol_target / \
                    (vol_row * close_row) 
            # position units = target dollar risk / instrument dollar volatility (signed by forecast)
            positions = np.nan_to_num(positions, nan=0, posinf=0, neginf=0)
            nominal_tot = np.linalg.norm(positions, close_row, ord=1)
            units_held.append(positions)
            weights = positions * close_row / nominal_tot
            weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
            weights_held.append(weights)

            portfolio_df.at[portfolio_i, "nominal"] = nominal_tot #total gross notional exposure.
            portfolio_df.at[portfolio_i , "leverage"] = nominal_tot / portfolio_df.at[portfolio_i, "capital"] #total leverage

            close_prev = close_row 
        
        return portfolio_df.set_index('datetime', drop=True)

    def zip_data_generator(self): #generator to save memory 
        for (portfolio_i, portfolio_row), \
            (ret_i, ret_row), (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (vol_i,vol_row) in zip(
                self.portfolio_df.iterrows(),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.voldf.iterrows() 
            ): #walk dfs together, one index at a time 
            yield {
                "portfolio_i": portfolio_i,
                "portfolio_row": portfolio_row,
                "ret_i": ret_i,
                "ret_row": ret_row,
                "close_row": close_row,
                "eligibles_row": eligibles_row,
                "vol_row":vol_row,
            }