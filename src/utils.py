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

def get_pnl_stats(last_weights, last_units, prev_close, ret_row, leverages):
    ret_row= np.nan_to_num(ret_row, nan=0, posinf=0,neginf=0)
    # Vectorized PnL: units * yesterday close gives dollar exposure per asset; multiply by return.
    day_pnl = np.sum(last_units * prev_close * ret_row) #dollar pnl 
    # Portfolio nominal return is weighted sum of asset returns.
    nominal_ret = np.dot(last_weights, ret_row) 
    capital_ret = nominal_ret * leverages[-1] #entire capital return is nominal_ret * prev day leverage (return on portfolio)
    return day_pnl, nominal_ret, capital_ret

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
        
        portfolio_df.at[0,"capital"]=10000
        portfolio_df.at[0,"day_pnl"]=0.0
        portfolio_df.at[0,"capital_ret"]=0.0
        portfolio_df.at[0,"nominal_ret"]=0.0

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
        
        # Build aligned matrix views once so downstream simulation can operate by date vector.
        closes, eligibles,vols, rets =[], [],[], []
        for inst in self.insts:
            df = pd.DataFrame(index = trade_range)
            inst_vol = (-1 + self.dfs[inst]['close']/self.dfs[inst]['close'].shift(1)).rolling(30).std() #simple vol using return, vol must be calculated before ffill as there will be duplicate in ffill
            #match indices to fill data (aval date <-> dfs[inst] of date)
            # align to trade_range and fill both directions like tutorial to avoid NaN leakage into vector math
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
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
        start = self.start + timedelta(hours=5)
        end = self.end + timedelta(hours=5)
        date_range = pd.date_range(start,end,freq="D") #trading date_range, not every ticker exist at pt of time
        self.compute_meta_info(trade_range=date_range)
        self.portfolio_df = self.init_portfolio_settings(trade_range = date_range)

        units_held, weights_held = [], []
        close_prev = None 
        # ewmas: EWMA of strategy variance proxy from capital_ret^2 (strategy-level risk, not per-asset vol)
        # ewstrats: EWMA of applied strategy scaling factor (strat_scalar) to smooth leverage changes
        ewmas , ewstrats = [0.01], [1] #default
        strat_scalars = [] #target_vol / realized_vol as scalar for position 
        capitals, nominal_rets, capital_rets = [10000.0], [0.0], [0.0]
        nominals, leverages = [], []

        for data in self.zip_data_generator():
            portfolio_i = data["portfolio_i"] 
            ret_i = data["ret_i"] 
            ret_row = data["ret_row"]
            close_row = data["close_row"]
            eligibles_row = data["eligibles_row"]
            vol_row = data["vol_row"]
            strat_scalar = 2

            if portfolio_i != 0:
                strat_scalar = self.get_strat_scaler(
                    target_vol = self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats = ewstrats #previous strat-scaler
                )
                day_pnl, nominal_ret, capital_ret = get_pnl_stats(last_weights=weights_held[-1], last_units=units_held[-1], prev_close=close_prev, 
                                                    ret_row=ret_row, leverages=leverages)
                
                capitals.append(capitals[-1] + day_pnl)
                nominal_rets.append(nominal_ret)
                capital_rets.append(capital_ret)
                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret !=0 else ewmas[-1]) # update EWMA variance estimate using squared portfolio return
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret !=0 else ewstrats[-1]) # update EWMA of scaling factor (not return volatility)

            strat_scalars.append(strat_scalar)
            forecasts = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            if type(forecasts) == pd.Series: forecasts = forecasts.values
            # Keep only tradable names for this date by zeroing non-eligible forecasts.
            forecasts = forecasts/eligibles_row
            forecasts = np.nan_to_num(forecasts,nan=0,posinf=0,neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))
            vol_target = (self.portfolio_vol / np.sqrt(253)) * capitals[-1] #annual_vol(in dollar) = daily_vol * capital
            positions = strat_scalar * \
                forecasts / forecast_chips \
                * vol_target  \
                / (vol_row * close_row) if forecast_chips != 0 else np.zeros(len(self.insts))
            # vol * close = dollar volatility per 1 unit; scaled forecast * vol_target = signed dollar risk target
            # position units = signed dollar risk target / dollar volatility per unit
            positions = np.nan_to_num(positions, nan=0, posinf=0, neginf=0)
            # Gross notional = L1 norm of dollar positions.
            nominal_tot = np.linalg.norm(positions * close_row, ord=1)
            units_held.append(positions)
            # Per-asset portfolio weights are dollar position divided by gross notional.
            weights = positions * close_row / nominal_tot
            weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
            weights_held.append(weights)

            nominals.append(nominal_tot)#total gross notional exposure.
            leverages.append(nominal_tot/capitals[-1]) #total leverage
            close_prev = close_row 

        # Rebuild full output frames from cached daily vectors.
        units_df = pd.DataFrame(data=units_held, index=date_range, columns=[inst + " units" for inst in self.insts])
        weights_df = pd.DataFrame(data=weights_held, index=date_range, columns=[inst + " w" for inst in self.insts])
        nom_ser = pd.Series(data=nominals, index=date_range, name="nominal_tot")
        lev_ser = pd.Series(data=leverages, index=date_range, name="leverages")
        cap_ser = pd.Series(data=capitals, index=date_range, name="capital")
        nomret_ser = pd.Series(data=nominal_rets, index=date_range, name="nominal_ret")
        capret_ser = pd.Series(data=capital_rets, index=date_range, name="capital_ret")
        scaler_ser = pd.Series(data=strat_scalars, index=date_range, name="strat_scalar")
        portfolio_df = pd.concat([
            units_df,
            weights_df,
            lev_ser,
            scaler_ser,
            nom_ser,
            nomret_ser,
            capret_ser,
            cap_ser
        ],axis=1)
        return portfolio_df

    def zip_data_generator(self): #yield aligned date rows across portfolio/ret/close/eligibility/vol frames
        for (portfolio_i), \
            (ret_i, ret_row), (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (vol_i,vol_row) in zip(
                range(len(self.retdf)),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.voldf.iterrows() 
            ): #walk dfs together, one index at a time 
            yield {
                "portfolio_i": portfolio_i,
                "ret_i": ret_i,
                "ret_row": ret_row.values,
                "close_row": close_row.values,
                "eligibles_row": eligibles_row.values,
                "vol_row":vol_row.values,
            }

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
