# -*- coding: utf-8 -*-
"""
Created on Thu Jun  14 13:56:34 2018

@author: HiddeFokkemaADC
"""

# -*- coding: utf-8 -*-
import pandas as pd
from pandas.io import sql
from sqlalchemy import create_engine
import glob
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
#Creating connecting with the sql database
engine = create_engine("mysql://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_raw?charset=utf8mb4")
con = engine.connect()

# Enforce UTF-8 for the connection.
engine.execute('SET NAMES utf8mb4')
engine.execute("SET CHARACTER SET utf8mb4")
engine.execute("SET character_set_connection=utf8mb4")

#Creating connecting with the sql database
engine_2 = create_engine("mysql://adc:ADC2018adc@cryptodb.c6fpuqrppv1v.us-east-1.rds.amazonaws.com/crypto_main?charset=utf8mb4")
con_2 = engine_2.connect()

# Enforce UTF-8 for the connection.
engine_2.execute('SET NAMES utf8mb4')
engine_2.execute("SET CHARACTER SET utf8mb4")
engine_2.execute("SET character_set_connection=utf8mb4")

models =  [['svm',SGDClassifier(loss='hinge',
                          penalty='l2',
                          alpha=1e-3,
                          max_iter=1000,
                          random_state=42)],

            ['log',SGDClassifier(loss='log',
                                      penalty='l2',
                                      alpha=1e-3,
                                      max_iter=1000,
                                      random_state=42)],

            ['NN', MLPClassifier(hidden_layer_sizes=(50, 50, 50))],

            ['Random Forest',RandomForestClassifier(n_estimators=100)]
]

#coin = input("What coin do you want to test? Only use symbols \n")

coins = {"BTC" : ["BTC", 'bitcoin' ],
                      "ETH" : ["ETH" ,"ethereum"],
                      "XRP": ["XRP", "ripple" ],
                      "BCH": ["BCH", "Bitcoin-cash" ],
                      "ADA": ["ADA", "cardano" ],
                      "LTC": ["LTC", "liteCoin" ],
                      "XEM": ["XEM", "nem" ],
                      "NEO": ["NEO", "neo" ],
                      "XLM": ["XLM", "stellar" ],
                      "EOS": ["EOS", "eos" ],
                      "MIOTA": ["MIOTA","iota" ],
                      #"DASH": ['"DASH" OR "Dash" '],
                      "XMR": ["XMR", "monero"],
                      "TRX": ["TRX", "tron" ],
                      "QASH": ["QASH", "QASH" ],
                      "BTG": ["BTG", "bitcoin-gold"],
                      "ICX": ["ICX", "icon" ],
                      "QTUM": ["QTUM", "qtum" ],
                      "ETC": ["ETC", "etheruem-classic"],
                      "LSK": ["LSK", "lisk" ],
                      #"NANO": ['"NANO"'],
                      "VEN": ["VEN", "vechain" ],
                     # "OMG": ['"OMG" OR "OmiseGO" '],
                      "PPT": ["PPT", "populous"],
                      "XVG": ["XVG", "verge" ],
                       "USDT": ["USDT", "tether" ],
}

#if coin not in coins.keys():
#    print("please pick a correct symbol")
#else:
#    symbol = coins[coin][0]
#    name = coins[coin][1]

symbol = "TRX"
name = "tron"
#time = int(input("What time delay do you want to look at? (min) \n"))
time_deltas = [ datetime.timedelta(minutes=15),
                datetime.timedelta(minutes=30),
                datetime.timedelta(minutes=45),
                datetime.timedelta(minutes=60)
                ]

timeperiod_1 = "2018-06-01 08:00:00"#input("From what starting time? format: YY-MM-DD hrs:min:sec \n" )
timeperiod_2 = "2018-06-19 20:00:00"#input("Till what end time? format: YY-MM-DD hrs:min:sec \n")

sql_1  = """SELECT AVG(score) as average,
	               sum(pos_words) as pos_words,
                   sum(neg_words) as neg_words,
                   count(symbol) as total,
                   symbol,
                   FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(created_at)/3600)*3600) AS timekey
             FROM crypto_raw.sentiment_per_tweet_new
             where symbol = '{}'
             and created_at between '{}' and '{}'
             group by timekey, symbol
             order by timekey desc, count(symbol) desc;
""".format(symbol, timeperiod_1, timeperiod_2)

sql_2 ="""  SELECT *
            from crypto_main.features_1h_5m
            where symbol = '{}'
            and entry_time between '{}' and '{}'
            order by entry_time asc;
""".format(symbol, timeperiod_1, timeperiod_2)

#feature_df = pd.read_sql(sql=sql_1, con=con)
feature_df = pd.read_sql(sql=sql_2, con=con_2)
feature  = feature_df.drop(labels=['entry_time', 'symbol'], axis=1).fillna(0)

print("Selecting {} data points".format(feature_df.shape[0]))
max_time = feature_df['entry_time'].max() + datetime.timedelta(hours=2)
min_time = feature_df['entry_time'].min() - datetime.timedelta(hours=2)

sql_price = """ SELECT reporting_date, price_eur, price_usd, price_btc
                FROM crypto_main.currency_hist_per_5_minute
                where currency_id = '{}'
                and reporting_date between '{}' and '{}'
                order by reporting_date asc;
""".format(name,min_time, max_time)
price_df = pd.read_sql(sql=sql_price, con=con_2)

for time_delta in time_deltas:
    time_df = feature_df['entry_time'] + time_delta
    price_df_temp = price_df.iloc[price_df['reporting_date'].searchsorted(time_df)]

    price_df_temp['change'] = price_df['price_eur'].diff().fillna(1)
    price_df_temp['change_bool'] = np.sign(price_df_temp['change'])

    X_train, X_test, y_train, y_test = train_test_split(feature, price_df_temp[['change_bool']], test_size=.25,
                                                random_state=42)

    print("start training")
    CV = 10
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model[0]
        accuracies = cross_val_score(model[1], feature, price_df_temp['change_bool'], scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    print("The results are")

    for model in models:
        change_clf = model[1].fit(X_train, y_train)
        predicted_change_svm = change_clf.predict(X_test)

        target_labels = np.unique(y_train['change_bool'].values)
        conf_mat_1 = confusion_matrix(y_test, predicted_change_svm, labels=target_labels)

        fig_1, ax_1 = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat_1, annot=True, fmt='d',
                    xticklabels=['decrease','same', 'increase'], yticklabels=['decrease','same', 'increase'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion matrix with a time delay of {} min and using {}'.format(int((time_delta.seconds)/60), model[0]))
        plt.savefig("Model_graphs\\conf_mat_{}min_{}_TRON.png".format(int((time_delta.seconds)/60), model[0]))

    fig_2, ax_2 = plt.subplots(figsize=(10,10))
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.title('Boxplot of CV scores with a time delay of {} min'.format(int((time_delta.seconds)/60)))
    plt.savefig("Model_graphs\\boxplot_CV_{}min_TRON.png".format(int((time_delta.seconds)/60)))

    plt.close("all")
