# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:02:49 2018

@author: SebastiaanHersmisADC
"""
import pandas as pd
from sqlalchemy import *
        
class Quotes(object):
    """Quotes is the main data loading class."""
    
    def import_data(currency_ids, start_date, end_date):
        
        #  import the data 
        engine = create_engine("mysql+mysqlconnector://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_raw")
        con = engine.connect()
        
        engine.echo = True
        metadata = MetaData(engine)
        
        # load 
        symbol_str = "('" + "','".join(currency_ids) + "')"
        
        # select only prices from top 5 coins
        sql = """SELECT currency_id,
                    price_usd,
                    reporting_date
                FROM crypto_main.currency_hist_per_5_minute
                WHERE currency_id in """ + symbol_str + """
                    AND reporting_date > '""" + start_date + """' 
                    AND reporting_date < '""" +  end_date  + """'
                ORDER BY currency_id ASC, reporting_date ASC"""
        
        # AND reporting_date > '""" + str(datetime.datetime.now() - datetime.timedelta(days = 30)) + """' 
        prices = pd.read_sql(sql, con)
        
        # forward fill
        prices = prices.fillna(method='ffill', limit=1)        
        
        # calculate relative multipliers of all combinations
        prices = pd.pivot_table(prices, index='reporting_date', columns='currency_id', values='price_usd')
    
        return prices               