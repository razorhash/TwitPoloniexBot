# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:56:51 2018

@author: SebastiaanHersmisADC
From https://www.quantstart.com/articles/My-Talk-At-The-London-Financial-Python-User-Group
"""

import pandas as pd
import os
os.chdir(r'C:\Users\SebastiaanHersmisADC\Documents\adc_crypto\041_Trading_structure')
         
from sqlalchemy import *

# import strategies
from strategies.pairTradingStrategy import PairTradingStrategy
from backtest import Backtest
from quotes import Quotes

from datetime import *
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/419163/
if __name__ == "__main__":

    # Create a Pair Trading Strategy instance 
    pairTradingStrategy = PairTradingStrategy(pair_id1 = "ripple", pair_id2 = "bitcoin",
                          MA_window = 60, band_with = 0.8)
    
    # backtest
    backtest = Backtest(start_date = "2018-03-27", end_date = "2018-03-28")  
    backtest_pairTradingStrategy = backtest.perform_backtest(pairTradingStrategy)
    
    # create report (Excel)
    backtest.create_report(backtest_pairTradingStrategy)
    
    # create graph
    backtest.create_graph(backtest_pairTradingStrategy)     
    