# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:58:10 2018

@author: SebastiaanHersmisADC
"""

import pandas as pd
import numpy as np
from strategy import Strategy
from orders import Orders
from datetime import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os 
os.chdir(r'C:\Users\SebastiaanHersmisADC\Documents\adc_crypto\041_Trading_structure')

# Pair trading strategy contains generate_signals(), output is used by the bot
class PairTradingStrategy(Strategy):
    """ Requires:
    pair_id1 and pair_id2 - The pair that is traded
    
    Optional:
    MA_window - Moving average window (in observations of 'data')
    band_with - number of std devs around SMA
    """
    
    def __init__(self, pair_id1, pair_id2, MA_window=100, band_with = 1):
        
        # set general variables
        self.pair_id1 = pair_id1
        self.pair_id2 = pair_id2 
        
        self.pair_id = pair_id1 + pair_id2    
        self.MA_window = MA_window
        self.band_with = band_with
        
    def generate_orders(self, quotes):
        """ Returns the DataFrame of symbols containing the signals
        to buy x or sell x: (x, -x). Note that the index (date) should
        be a subset of the full 'data' vector """
                     
        ## --- Prepare data ---------------------------------------------------
        quotes = self.enrich_data(quotes)
        
        # possibly add more data
        data = quotes
        
        ## ---- Perform strategy ----------------------------------------------
        # moving average of 1 day = 288 obs, 6 hours = 72 obs
        SMA_fill = np.full(self.MA_window-1, np.nan)
        SMA = np.convolve(data[self.pair_id].as_matrix(), np.ones((self.MA_window,))/self.MA_window, mode='valid')
        SMA = np.concatenate((SMA_fill, SMA), axis=0)
        
        # upper and lower bound
        std_upper = SMA + self.band_with * np.std(data[self.pair_id].as_matrix())
        std_lower = SMA - self.band_with * np.std(data[self.pair_id].as_matrix())
        
        # see where we breach the boundary
        sell = -1 * (std_upper < data[self.pair_id])
        # only include first breach before return
        sell = np.append(0, np.diff(sell))
        sell[sell > 0] = 0
        
        # see where we breach the boundary        
        buy = 1 * (std_lower > data[self.pair_id])  
        # only include first breach before return
        buy = np.append(0, np.diff(buy))
        buy[buy < 0] = 0
            
        # initiate vector and convert to correct format
        signals = pd.DataFrame(index = data.index) 
        signals["signal"] = sell + buy      
        signals["currency_id"] = np.nan
        
        # filter to actions
        signals = signals[signals.signal != 0]          
        
        # buy means buy self.pair_id1 and sell self.pair_id2
        buy_pair = signals[signals.signal == 1]
        buy_signals = pd.DataFrame({"currency_id": self.pair_id1, "signal": 1}, index = buy_pair.index)        
        buy_signals = buy_signals.append(pd.DataFrame({"currency_id": self.pair_id2, "signal": -1}, index = buy_pair.index))
        
        # sell
        sell_pair = signals[signals.signal == -1]
        sell_signals = pd.DataFrame({"currency_id": self.pair_id1, "signal": -1}, index = sell_pair.index)
        sell_signals = sell_signals.append(pd.DataFrame({"currency_id": self.pair_id2, "signal": 1}, index = sell_pair.index))
        
        # combine 
        signals = pd.concat([buy_signals, sell_signals])

        # return sell_pair
        signals = signals.sort_index()
        
        return Orders(signals)

    def currency_ids(self):
        """ Every trading strategy shioulf have something like this: return a vector with the currencies used"""
        return [self.pair_id1, self.pair_id2]
    
    def research(self, data):
        """ Perform research, using the data frame"""
        
        ## ---- Calculate relative (extended) prices --------------------------
        Extended_prices = self.enrich_data(data)
        
        ## ---- Plot all ------------------------------------------------------
        for column in Extended_prices:
            
            # ...
            fig, ax = plt.subplots(figsize=(16, 10))   
            ax.plot(pd.to_datetime(Extended_prices.index), Extended_prices[column].as_matrix())
            fig.autofmt_xdate()
            
            # add title and format axes
            ax.set_title(str(column))    
            ax.grid(which='both')
            ax.minorticks_on()
            
            # moving average of 1 day = 288 obs, 6 hours = 72 obs
            MA_window = 60
            band_with = 0.8
            SMA_fill = np.full(MA_window-1, np.nan)
            SMA = np.convolve(Extended_prices[column].as_matrix(), np.ones((MA_window,))/MA_window, mode='valid')
            SMA = np.concatenate((SMA_fill, SMA), axis=0)
            
            # upper and lower bound
            std_upper = SMA + band_with * np.std(Extended_prices[column].as_matrix())
            std_lower = SMA - band_with * np.std(Extended_prices[column].as_matrix())
            
            # see where we breach the boundary
            sell = -1 * (std_upper < Extended_prices[column])
            # only include first breach before return
            sell = np.append(0, np.diff(sell))
            # https://stackoverflow.com/questions/27778299/
            sell = sell.astype('float')
            sell[sell >= 0] = np.nan    
            sell = -sell * Extended_prices[column]
            
            # see where we breach the boundary        
            buy = 1 * (std_lower > Extended_prices[column])  
            # only include first breach before return
            buy = np.append(0, np.diff(buy))
            # https://stackoverflow.com/questions/27778299/
            buy = buy.astype('float')
            buy[buy <= 0] = np.nan
            buy = buy * Extended_prices[column]
                
            # add SMA bounds to graph
            ax.plot(pd.to_datetime(Extended_prices.index), std_upper, color='grey')
            ax.plot(pd.to_datetime(Extended_prices.index), std_lower, color='grey')
            
            # add buys and sells
            ax.scatter(pd.to_datetime(Extended_prices.index), buy, color='green', s = 100, facecolors='none', linewidth=2.5)
            ax.scatter(pd.to_datetime(Extended_prices.index), sell, color='red', s = 100, facecolors='none',  linewidth=2.5)      
            
            os.chdir(r'C:\Users\SebastiaanHersmisADC\Documents\adc_crypto\022_Data analysis\Technical')
            
            plt.savefig('Pairs\\' + column + '.png')
            plt.close("all")
            
    def enrich_data(self, data):
        
        ## ---- calculate relative (extended) prices --------------------------
        Extended_prices = data
        for column in data:
            # differences of Prices with column
            Diff_prices = (data.div(data[column].values,axis=0))
            Diff_prices.columns = [str(col) + column for col in Diff_prices.columns]
            
            # append Diff_prices to Extended prices
            Extended_prices = pd.concat([Extended_prices, Diff_prices], axis=1, join="inner")
        
        return Extended_prices