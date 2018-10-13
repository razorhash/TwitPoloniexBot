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

class Portfolio(object):
    """ Contains elements of the portfolio, such as backtesting"""
    
    def __init__(self, signals = pd.DataFrame(), initial_capital=100000, 
                 start_date = None, end_date = None, strategy_name = "Unknown"):
        self.signals = signals
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name

    def backtest(self, strategy):
        """ Perform backtest of strategy 'strategy' """ 
        
        # load data
        data = self.import_data(currency_ids = np.unique(self.signals.currency_id),
                                start_date = self.start_date,
                                end_date = self.end_date)
            
        # add cash and columns for currencies
        data["capital"] = 0
        data["actions"] = "" # should become a dictW        
        data["description"] = ""
        data["capital"].iloc[0] = self.initial_capital
        data["worth_usd"] = 0
        
        for currency_id in np.unique(self.signals.currency_id):
            data[currency_id + '_position'] = 0
        
        # a sell or buy can influence subsequent positions, so calculate iteratively
        for observation in range(1, len(data.index)):
            
            date = data.index[observation]
            
            # investment this period is zero
            investment_capital_period = 0
            
            # amount of currency_ids initially same as last period
            for currency_id in np.unique(self.signals.currency_id):
                data[currency_id + '_position'].iloc[observation] = data[currency_id + '_position'].iloc[observation-1]                    
          
            # at each point, compute size of each position (cash and currencies), and record actions
            if(data.index[observation] in self.signals.index):
                                
                action_df = pd.DataFrame(columns=list(["Currency","NominalAmount", "CapitalAmount"]))
                
                # could be multiple actions
                for index, action in self.signals.loc[date].iterrows():    
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
        return data
    
    def create_report(self, backtest):
        """ Export the backtesting results to Excel file. 
        'Backtest' contains the testing results"""
        
        writer = pd.ExcelWriter('backtest_results/' + self.strategy_name + '_' + str(date.today()) + '.xlsx')
        backtest.to_excel(writer,'Backtest result')
        writer.save()
        
        print('saved')
        return true
    
    def create_graph(self, backtest):
        """ Display the backtesting result in a graph. 
        'Backtest' contains the testing results
        Graph should contain: 
            - graph of price of each currency mentioned in the results, including buy/sell moments
            - amount of capital invested in cash and each currency
            - """
        
        # check number of currencies
        no_currencies = len(np.unique(signals.currency_id))
        
        # plot capital
        fig, ax = plt.subplots(1+no_currencies/2,2)
        ax[0, 0].plot(backtest.capital)
        ax[0, 1].plot(backtest.capital) 
        
        # plot each currency 
        
        return true
    
    def import_data(self, currency_ids = [], start_date = None, end_date = None):
        
        # check start and end date
        if(self.start_date is None):
            self.start_date = "1900-01-01"
            
        if(self.end_date is None):
            self.end_date = str(date.today())
            
        import pandas as pd
        #  import the data 
        engine = create_engine("mysql+mysqlconnector://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_raw")
        con = engine.connect()
        
        engine.echo = True
        metadata = MetaData(engine)
        
        # load 
        symbol_str = "('" + "','".join(currency_ids.tolist()) + "')"
        
        # select only prices from top 5 coins
        sql = """SELECT currency_id,
                    price_usd,
                    reporting_date
                FROM crypto_main.currency_hist_per_5_minute
                WHERE currency_id in """ + symbol_str + """
                    AND reporting_date > '""" + self.start_date + """' 
                    AND reporting_date < '""" +  self.end_date  + """'
                ORDER BY currency_id ASC, reporting_date ASC"""
        
        # AND reporting_date > '""" + str(datetime.datetime.now() - datetime.timedelta(days = 30)) + """' 
        Prices = pd.read_sql(sql, con)
        
        # forward fill
        Prices = Prices.fillna(method='ffill', limit=1)        
        
        # calculate relative multipliers of all combinations
        Prices = pd.pivot_table(Prices, index='reporting_date', columns='currency_id', values='price_usd')
    
        # create index
        # Prices.index = Prices.reporting_date
        # del Prices['reporting_date']
        return Prices
        