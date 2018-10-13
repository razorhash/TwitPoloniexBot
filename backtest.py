# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:20:39 2018
https://www.quantstart.com/articles/My-Talk-At-The-London-Financial-Python-User-Group
@author: SebastiaanHersmisADC
"""
import pandas as pd
import numpy as np
from datetime import *
from sqlalchemy import *
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from quotes import Quotes
from orders import Orders

class Backtest(object):
    """ Contains the backtesting object, such as perform_backtest"""
    
    def __init__(self, start_date, end_date, initial_capital=100000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

    def perform_backtest(self, strategy):
        """ Perform backtest of strategy (class: Strategy)""" 
            
        # 1 ---- load quotes data ---------------------------------------------
        # get relevant coins
        currency_ids = strategy.currency_ids()
        
        data = Quotes.import_data(currency_ids = currency_ids,
                                  start_date = self.start_date,
                                  end_date = self.end_date)
            
        # 2 ---- use generate_orders() to generate orders ---------------------        
        strategy_orders = strategy.generate_orders(data)

        if not isinstance(strategy_orders, Orders):
            raise TypeError("Orders is not of class 'orders'")        
        
        print(strategy_orders)
        strategy_orders = strategy_orders.get_orders()
        
        # 3 ---- perform backtest ---------------------------------------------
        
        # add cash and columns for currencies
        data["capital"] = 0
        data["actions"] = ""      
        data["description"] = ""
        data["capital"].iloc[0] = self.initial_capital
        data["worth_usd"] = 0
        
        for currency_id in currency_ids:
            data[currency_id + '_position'] = 0
        
        # a sell or buy can influence subsequent positions, so calculate iteratively
        for observation in range(1, len(data.index)):
            
            date = data.index[observation]
            
            print(date)
            
            # investment this period is zero
            investment_capital_period = 0
            
            # amount of currency_ids initially same as last period
            for currency_id in currency_ids:
                data[currency_id + '_position'].iloc[observation] = data[currency_id + '_position'].iloc[observation-1]                    
          
            # at each point, compute size of each position (cash and currencies), and record actions
            if(data.index[observation] in strategy_orders.index):
                                
                action_df = pd.DataFrame(columns=list(["Currency","NominalAmount", "CapitalAmount"]))
                
                # could be multiple actions
                for index, action in strategy_orders.loc[date].iterrows():    
                    currency_id = action['currency_id']
                    signal = action['signal']
                    
                    # Buy
                    if signal == 1:
                        
                        # buy for 10% currency_id
                        investment_capital = data["capital"].iloc[observation-1] * 0.10   

                        # estimate how many coins
                        investment_nominal = round(investment_capital / data[currency_id].iloc[observation])
                        
                        # calculate exact capital needed
                        investment_capital_exact = investment_nominal * data[currency_id].iloc[observation]
                        investment_capital_period = investment_capital_period + investment_capital_exact 
                        
                        # change the amount of currency hold
                        data[currency_id + '_position'].iloc[observation] = data[currency_id + '_position'].iloc[observation-1] + investment_nominal
                        
                        # report action by appending a Series to the (empty) dataframe
                        action_df = action_df.append(pd.Series({"Currency": currency_id, 
                                                                   "NominalAmount": investment_nominal, 
                                                                   "CapitalAmount": investment_capital_exact}),ignore_index=True)
                        
                        # report description
                        data["description"].iloc[observation] = (data["actions"].iloc[observation] + "\n Buy " + 
                                                            str(investment_nominal) + " " + str(currency_id) + 
                                                            " for " + str(investment_capital_exact))
                    
                    # Sell
                    if signal == -1:
                        
                        # sell currency_id for 10% of total capital
                        investment_capital = data["capital"].iloc[observation-1] * 0.10   
                        
                        # estimate how many coins
                        investment_nominal = round(investment_capital / data[currency_id].iloc[observation])
                        
                        # calculate exact capital needed
                        investment_capital_exact = investment_nominal * data[currency_id].iloc[observation]
                        investment_capital_period = investment_capital_period - investment_capital_exact
                        
                        # change the amount of currency hold
                        data[currency_id + '_position'].iloc[observation] = data[currency_id + '_position'].iloc[observation-1] - investment_nominal
                                                
                        # report action
                        action_df = action_df.append(pd.Series({"Currency": currency_id, 
                                                                   "NominalAmount": investment_nominal, 
                                                                   "CapitalAmount": investment_capital_exact}),ignore_index=True)
                                           
                        # report description
                        data["description"].iloc[observation] = data["actions"].iloc[observation] + "Sell " + str(investment_nominal) + " " + str(currency_id) + " for " + str(investment_capital_exact)
                 
                # report actions
                data["actions"].iloc[observation] = action_df.to_json()
                
            # calculate resulting cash capital
            data["capital"].iloc[observation] = data["capital"].iloc[observation-1] - investment_capital_period
            
            # calculate worth by capital (usd) and each currency * price
            data["worth_usd"].iloc[observation] = data["capital"].iloc[observation]
            
        # return a backtest dict, which is a dict
        return {"backtest_df": data, "strategy": strategy}
            
    def create_report(self, backtest):
        """ Export the backtesting results to Excel file. 
        'Backtest' contains the testing results"""
        
        backtest_results = backtest['backtest_df']
        strategy = backtest['strategy']
        
        # https://stackoverflow.com/questions/510972/
        strategy_name = strategy.__class__.__name__
        
        writer = pd.ExcelWriter('backtest_results/' + strategy_name + '_' + str(date.today()) + '.xlsx')
        backtest_results.to_excel(writer,'Backtest result')
        
        writer.save()
        
        print('saved')
        return True
    
    def create_graph(self, backtest):
        """ Display the backtesting result in a graph. 
        'Backtest' contains the testing results
        Graph should contain: 
            - graph of price of each currency mentioned in the results, including buy/sell moments
            - amount of capital invested in cash and each currency
            - """
        
        backtest_results = backtest['backtest_df']
        strategy = backtest['strategy']
        
        # check number of currencies
        no_currencies = len(strategy.currency_ids())
        
        fig, ax = plt.subplots(round(1+no_currencies/2),2)
        
        # plot capital
        ax[0, 0].plot(backtest_results.capital)        
        # plot total worth in USD
        ax[0, 1].plot(backtest_results.worth_usd) 
      
        # plot total value of each currency 
        i = 1
        for currency_id in strategy.currency_ids:
            ax[round(i/2), i % 2].plot(backtest_results.capital)   
            i = i + 1
        
        return True