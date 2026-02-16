import pandas as pd 
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime 
from io import StringIO
import pytz 
import yfinance as yf
import threading #python multi-threading
from utils import load_pickle, save_pickle
from utils import Alpha
from alpha1 import Alpha1
from alpha2 import Alpha2
from alpha3 import Alpha3
import matplotlib.pyplot as plt
from utils import Portfolio
from utils import timeme
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

start = datetime(2010, 1, 1, tzinfo=pytz.utc) #standardized tz
end = datetime.now(pytz.utc)

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')

    #defensive block
    if not tables:
        raise ValueError('No table found')
    table = tables[0]
    dfs = pd.read_html(StringIO(str(table)))
    if not dfs:
        raise ValueError('No tables in selected html')
    
    df = dfs[0]
    tickers = df['Symbol'].tolist()
    tickers = [s.replace('.', '-') for s in tickers]
    return tickers 

tickers = get_sp500_tickers()

def get_history(ticker, start, end, granularity='1d', tries=9):
    try:
        df = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval=granularity,
            auto_adjust=True,
        ).reset_index() #minus dividend etc...
    except Exception as err:
        if tries < 5:
            return get_history(ticker, start, end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"})
    
    #defensive block
    if df.empty: 
        return pd.DataFrame() #return empty if no df
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize(pytz.utc)
    else:
        df['datetime'] = df['datetime'].dt.tz_convert(pytz.utc)

    df = df.drop(columns=['Dividends','Stock Splits'])
    df = df.set_index('datetime',drop=True)
    return df

def get_histories(tickers, start, end, granularity='1d'):
    dfs = [None] * len(tickers) #list of all dfs set to None
    def _helper(i):
        print(tickers[i])
        df = get_history(tickers[i], start[i], end[i], granularity=granularity)
        dfs[i]= df #assign i-th df to i-th spot

    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))] #python multi-thread
    [thread.start() for thread in threads] 
    [thread.join() for thread in threads]

    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty] #filter out non-exisitng ticker 
    dfs = [df for df in dfs if not df.empty] #filter out non-exisitng ticker's df (empty)
    return tickers, dfs 

def get_tickers_dfs(start,end): 
    try:
        tickers,ticker_dfs = load_pickle('dataset.obj')
    except Exception as err:
        tickers = get_sp500_tickers()
        starts = [start]*len(tickers)
        ends = [end]*len(tickers)
        tickers, dfs = get_histories(tickers, starts,ends, granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers, dfs)} #dict of ticker(key):df(value) 
        save_pickle('dataset.obj', (tickers, ticker_dfs))
    return tickers, ticker_dfs

def main():
    tickers, ticker_dfs = get_tickers_dfs(start=start, end=end)
    # testfor = 200
    # print(f'Testing {testfor} of out {len(tickers)} tickers')
    # tickers = tickers[:testfor]

    alpha1 = Alpha1(insts=tickers, dfs=ticker_dfs,start=start,end=end)
    alpha2 = Alpha2(insts=tickers, dfs=ticker_dfs,start=start,end=end)
    alpha3 = Alpha3(insts=tickers, dfs=ticker_dfs,start=start,end=end)
    df1 = alpha1.run_simulation()
    df2 = alpha2.run_simulation()
    df3 = alpha3.run_simulation()

    print(list(df1.capital)[-1])
    print(list(df2.capital)[-1])
    print(list(df3.capital)[-1])

if __name__ == "__main__":
    main()
