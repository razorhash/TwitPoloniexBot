# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:29:45 2018

@author: HiddeFokkemaADC
"""

# -*- coding: utf-8 -*-
import pandas as pd
from pandas.io import sql
from sqlalchemy import create_engine
import glob
from nltk import tokenize
import datetime
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
#Creating connecting with the sql database
engine = create_engine("mysql://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_raw?charset=utf8mb4")
con = engine.connect()

# Enforce UTF-8 for the connection.
engine.execute('SET NAMES utf8mb4')
engine.execute("SET CHARACTER SET utf8mb4")
engine.execute("SET character_set_connection=utf8mb4")

#Creating connecting with the sql database
engine_2 = create_engine("mysql://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_raw?charset=utf8mb4")
con_2 = engine_2.connect()

# Enforce UTF-8 for the connection.
engine_2.execute('SET NAMES utf8mb4')
engine_2.execute("SET CHARACTER SET utf8mb4")
engine_2.execute("SET character_set_connection=utf8mb4")

coin = "TRX"
market_tag = 'tron'

sql_1  = """SELECT avg(count), symbol
            FROM(
		          select 	count(symbol) as count,
				            symbol as symbol,
                            FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(created_at)/300)*300) AS timekey
		          from crypto_raw.sentiment_per_tweet_new
                  where symbol = 'TRX'
		          group by timekey
	        ) as average;
"""

sql_2 = """
SELECT count(symbol) as total,
    FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(created_at)/300)*300) AS timekey
FROM crypto_raw.sentiment_per_tweet_new
WHERE symbol = 'TRX'
group by timekey
order by timekey desc;
"""

total_tweets = pd.read_sql(sql_2, con=con)
average = pd.read_sql(sql_1, con=con)
total_tweets = total_tweets.iloc[1:]

total_tweets['total'] = total_tweets['total'] - average['avg(count)'][0]
max_time = total_tweets['timekey'].max()
min_time = total_tweets['timekey'].min()
total_tweets = total_tweets.sort_values('timekey')

sql_price = """ SELECT reporting_date, price_eur, price_usd, price_btc
                FROM crypto_main.currency_hist_per_5_minute
                where currency_id = 'tron'
                and reporting_date between '{}' and '{}'
                order by reporting_date desc;
""".format(min_time, max_time)

prices = pd.read_sql(sql=sql_price, con=con_2)
prices = prices.sort_values('reporting_date')

start_eur = 10000
start_trx = 0
prev = 0
index = 0
money = []
trx = []
short = 0
for row in total_tweets.itertuples():
    if prices[prices['reporting_date'] < row[2]]['price_eur'].empty:
        continue
    else:
        if row[1] > 0 and row[1] * prev < 0:
            start_eur -= (short*2000 + 2000) * prices[prices['reporting_date'] < row[2]]['price_eur'].iloc[-1]
            start_trx += 2000
            short = 0
            money.append(start_eur)
        elif row[1] < 0 and row[1] * prev < 0:
            start_eur += (start_trx  + 2000)* prices[prices['reporting_date'] < row[2]]['price_eur'].iloc[-1]
            start_trx = 0
            short += 1
            money.append(start_eur)
    prev = row[1]
if start_trx != 0:
    start_eur += start_trx * prices['price_eur'].iloc[-1]
    money.append(start_eur)
print('started with 10000 euros \n after trading we have {} euros'.format(start_eur))
plt.plot( money)
plt.show()
