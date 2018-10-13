# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:56:51 2018

@author: SebastiaanHersmisADC
From https://www.quantstart.com/articles/My-Talk-At-The-London-Financial-Python-User-Group
"""

import pandas as pd
import os
os.chdir(r'C:\Users\SebastiaanHersmisADC\Documents\adc_crypto\031_Model exploration')
         
# import strategies
from strategies.pairTradingStrategy import PairTradingStrategy
from portfolio import Portfolio

from sqlalchemy import *
from datetime import *

import json
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/419163/
if __name__ == "__main__":
    
    ## ---- Obtain data: TODO should be put in a class 'load' or something...
    engine = create_engine("mysql+mysqlconnector://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_raw")
    con = engine.connect()
    
    engine.echo = True
    metadata = MetaData(engine)
    
    # load Dict and Price table: https://stackoverflow.com/questions/12047193/
    sql = "SELECT DISTINCT(id), rank FROM crypto_raw.CoinMarketCapTicker_old LIMIT 0, 20;"
    Dict = pd.read_sql(sql, con)
    symbol_str = "('" + "','".join(Dict['id'].values.tolist()) + "')"
    
    # select only prices from top 5 coins
    sql = """SELECT currency_id,
                price_btc,
                price_usd,
                reporting_date
            FROM crypto_main.currency_hist_per_5_minute
            WHERE currency_id in """ + symbol_str + """
                AND reporting_date > '""" + str(datetime.now() - timedelta(days = 30)) + """' 
            ORDER BY currency_id ASC, reporting_date ASC"""
    
    # AND reporting_date > '""" + str(datetime.datetime.now() - datetime.timedelta(days = 30)) + """' 
    Prices = pd.read_sql(sql, con)
    
    # forward fill
    Prices = Prices.fillna(method='ffill', limit=1)        
    
    # create index
    Prices.index = Prices.reporting_date
    del Prices['reporting_date']
    
    # calculate relative multipliers of all combinations
    Prices = pd.pivot_table(Prices, index='reporting_date', columns='currency_id', values='price_usd')
    
    relative_prices = Prices
    for column in Prices:
        # differences of Prices with column
        Diff_prices = (Prices.div(Prices[column].values,axis=0))
        Diff_prices.columns = [str(col) + column for col in Diff_prices.columns]
    
        # append Diff_prices to Extended prices
        relative_prices = pd.concat([relative_prices, Diff_prices], axis=1, join="inner")
    
    data = relative_prices # pd.concat([Prices, relative_prices], axis=1) 

    ## --- Load until here

    # Create a Pair Trading Strategy instance 
    pair = PairTradingStrategy("ripple", "bitcoin", data,
                          MA_window = 60, band_with = 0.8)
    signals = pair.generate_signals()

    # initiate a portfolio with $100,000 initial capital
    portfolio = Portfolio(signals, initial_capital=100000, strategy_name = "pairtrading")
    
    # backtest!
    backtest = portfolio.backtest()  
    
    # create report (Excel)
    portfolio.create_report(backtest)
    
    # create graph
    portfolio.create_graph(backtest)
    
    print(returns.loc[returns.actions != ""].actions)
     
    