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
import copy
from collections import defaultdict
import matplotlib as mpl
import matplotlib.dates as mdates
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

pd.options.mode.chained_assignment = None
mpl.style.use('seaborn')
## TODO training en testing implementeren. Strategie implemeneteren en plotten

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

class AlgorithmTrading:
    '''
    Class to perform AlgorithmTrading on a set of coins of your choice and to test the performance.
    Every Trading class has the same models and coins that can be used. These lists can of course
    be expanded.
    '''

    def __init__(self, coins, ratios, money, model, lag=60, timeperiod_1="2018-06-25 08:00:00", timeperiod_2 =  "2018-07-03 20:00:00"):
        """
        Initialize the class by defining which coins you want to use in your portfolio, what ratios they have, how much money is in the portfolio
        , which model you want to use, what time lag you want to use and in which period you want to look. The training is done until the second to last day.
        The testing is then done on the last day. This is done to replicate the general strategy used by the bot, training on history to predict the future
        """
        self.models = {'svm': SGDClassifier(loss='hinge',
                                  penalty='l2',
                                  alpha=1e-3,
                                  max_iter=1000,
                                  random_state=42),

						'log' : SGDClassifier(loss='log',
                                              penalty='l2',
                                              alpha=1e-3,
                                              max_iter=1000,
                                              random_state=42),

						'NN' : MLPClassifier(hidden_layer_sizes=(50, 50, 50)),

						'Random Forest' : RandomForestClassifier(n_estimators=100),

						'Naive Bayes Gauss': GaussianNB(priors=[0.5, 0, 0.5])
        }

        #Coin symbols and name, to be used in the sql queries.
        self.coin_dict =  {   "BTC" : ["BTC", 'bitcoin'],
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
                              "USDT": ["USDT", "tether" ]
        }

        if model in self.models.keys():
            self.model_name = model
            self.model = self.models[model]
        else:
            raise ValueError("Model name not present. Choose from {}".format(self.models.keys()))

        if isinstance(coins, list) and isinstance(ratios, list) and sum(ratios) == 1:
            self.coins_ratios = list(zip(coins, ratios))
        elif not (isinstance(coins, list) and isinstance(ratios, list)):
            raise ValueError("Coins and ratios must be list")
        elif sum(ratios) != 1:
            raise ValueError("Ratios must add up to 1.")

        self.lag = datetime.timedelta(minutes=lag)
        self.money = money
        self.timeperiod_1 = timeperiod_1
        self.timeperiod_2 = (datetime.datetime.strptime(timeperiod_2, "%Y-%m-%d %X") - datetime.timedelta(days=1)).strftime("%Y-%m-%d %X")
        self.date_test = timeperiod_2

    def trading_train(self, query ='*'):
        """
        Function that will train the model on the data for the coins that were specified.
        """
        self.coin_data = []
        self.coin_features = []
        self.price_data = []
        self.trained_models = {}
        for i, (coin, _ ) in enumerate(self.coins_ratios):
            sql_features = """  SELECT {}
                        from crypto_main.features_1h_5m
                        where symbol = '{}'
                        and entry_time between '{}' and '{}'
                        order by entry_time asc;
            """.format(query, self.coin_dict[coin][0], self.timeperiod_1, self.timeperiod_2)

            self.coin_data.append(pd.read_sql(sql=sql_features, con=con_2))
            self.coin_features.append(self.coin_data[i].drop(labels=['entry_time', 'symbol'], axis=1).fillna(0))

            min_time = self.coin_data[i]['entry_time'].min() - datetime.timedelta(hours=2)
            max_time = self.coin_data[i]['entry_time'].max() + datetime.timedelta(hours=2)

            sql_price = """ SELECT reporting_date, price_eur, price_usd, price_btc
                            FROM crypto_main.currency_hist_per_5_minute
                            where currency_id = '{}'
                            and reporting_date between '{}' and '{}'
                            order by reporting_date asc;
            """.format(self.coin_dict[coin][1], min_time, max_time)

            price_df = pd.read_sql(sql=sql_price, con=con_2)
            time_df = self.coin_data[i]['entry_time'] + self.lag
            price_df_temp = price_df.iloc[price_df['reporting_date'].searchsorted(time_df)]
            self.price_data.append(price_df_temp)

            price_df_temp['change'] = price_df['price_eur'].diff().fillna(1)
            price_df_temp['change_bool'] = np.sign(price_df_temp['change'])

            X_train, _, y_train, _ = train_test_split(self.coin_features[i], price_df_temp['change_bool'], test_size=0,
                                                        random_state=42)

            model_temp = copy.copy(self.model)
            self.trained_models[coin] =  copy.copy(model_temp.fit(X_train, y_train))

    def plot_confusion_matrix(self, query ='*'):
        """
        Plot a confusion matrix to see how the model is performing. Class must have been trained before this can be used.
        """
        if not self.trained_models.keys():
            raise AttributeError("Models not trained. First call trading_train()")

        for i, (coin, _ ) in enumerate(self.coins_ratios):
            sql_features = """  SELECT {}
                        from crypto_main.features_1h_5m
                        where symbol = '{}'
                        and entry_time between '{}' and '{}'
                        order by entry_time asc;
            """.format(query, self.coin_dict[coin][0], self.timeperiod_2, self.date_test)

            test_data = pd.read_sql(sql=sql_features, con=con_2)
            test_features = (test_data.drop(labels=['entry_time', 'symbol'], axis=1)).fillna(0)

            min_time = test_data['entry_time'].min() - datetime.timedelta(hours=2)
            max_time = test_data['entry_time'].max() + datetime.timedelta(hours=2)

            sql_price = """ SELECT reporting_date, price_eur, price_usd, price_btc
                            FROM crypto_main.currency_hist_per_5_minute
                            where currency_id = '{}'
                            and reporting_date between '{}' and '{}'
                            order by reporting_date asc;
            """.format(self.coin_dict[coin][1], min_time, max_time)

            price_df = pd.read_sql(sql=sql_price, con=con_2)

            time_df = test_data['entry_time'] + self.lag
            price_df_temp = price_df.iloc[price_df['reporting_date'].searchsorted(time_df)]

            price_df_temp['change'] = price_df['price_eur'].diff().fillna(1)
            price_df_temp['change_bool'] = np.sign(price_df_temp['change'])

            target_labels = np.unique(price_df_temp['change_bool'].values)

            predicted_change = self.trained_models[coin].predict(test_features)
            conf_mat_1 = confusion_matrix(price_df_temp['change_bool'], predicted_change, labels=target_labels)

            fig_1, ax_1 = plt.subplots(figsize=(10,10))
            sns.heatmap(conf_mat_1, annot=True, fmt='d',
                        xticklabels=['decrease','same', 'increase'], yticklabels=['decrease','same', 'increase'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion matrix of {} with a time delay of {} min and using {}'.format(coin, int((self.lag.seconds)/60), self.model_name))
            return fig_1
            #plt.show()

    def perform_CV(self, CV = 5, query ='*'):
        """
        Cross validation is performed on all models for the coins and scores are plotted.
        Class must be trained before this function can be used
        """
        if not self.trained_models.keys():
            raise AttributeError("Models not trained. First call trading_train()")

        for i, (coin, _ ) in enumerate(self.coins_ratios):
            sql_features = """  SELECT {}
                        from crypto_main.features_1h_5m
                        where symbol = '{}'
                        and entry_time between '{}' and '{}'
                        order by entry_time asc;
            """.format(query, self.coin_dict[coin][0], self.timeperiod_2, self.date_test)
            test_data = pd.read_sql(sql=sql_features, con=con_2)
            test_features = (test_data.drop(labels=['entry_time', 'symbol'], axis=1)).fillna(0)
            min_time = test_data['entry_time'].min() - datetime.timedelta(hours=2)
            max_time = test_data['entry_time'].max() + datetime.timedelta(hours=2)

            sql_price = """ SELECT reporting_date, price_eur, price_usd, price_btc
                            FROM crypto_main.currency_hist_per_5_minute
                            where currency_id = '{}'
                            and reporting_date between '{}' and '{}'
                            order by reporting_date asc;
            """.format(self.coin_dict[coin][1], min_time, max_time)
            price_df = pd.read_sql(sql=sql_price, con=con_2)
            time_df = test_data['entry_time'] + self.lag
            price_df_temp = price_df.iloc[price_df['reporting_date'].searchsorted(time_df)]
            price_df_temp['change'] = price_df['price_eur'].diff().fillna(1)
            price_df_temp['change_bool'] = np.sign(price_df_temp['change'])

            cv_df = pd.DataFrame(index=range(CV * len(self.models)))
            entries = []
            for model in self.models:
                model_name = model
                accuracies = cross_val_score(self.models[model], test_features, price_df_temp['change_bool'], scoring='accuracy', cv=CV)
                for fold_idx, accuracy in enumerate(accuracies):
                    entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

            fig_1, ax_1 = plt.subplots(figsize=(10,10))
            sns.boxplot(x='model_name', y='accuracy', data=cv_df)
            sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                          size=8, jitter=True, edgecolor="gray", linewidth=2)
            plt.title('Boxplot of CV scores of {} with a time delay of {} min'.format(coin, int((self.lag.seconds)/60)))
            return fig_1
            #plt.show()

    def trading_test(self, plot=True, query ='*'):
        if not self.trained_models.keys():
            raise AttributeError("Models not trained. First call trading_train()")

        predictions = []
        dict_money = {}
        prices_coins = []
        for i, (coin, ratio ) in enumerate(self.coins_ratios):
            dict_money[coin] = ratio*self.money
            sql_features = """  SELECT {}
                        from crypto_main.features_1h_5m
                        where symbol = '{}'
                        and entry_time between '{}' and '{}'
                        order by entry_time asc;
            """.format(query, self.coin_dict[coin][0], self.timeperiod_2, self.date_test)

            test_data = pd.read_sql(sql=sql_features, con=con_2)
            test_features = (test_data.drop(labels=['entry_time', 'symbol'], axis=1)).fillna(0)
            min_time = test_data['entry_time'].min() - datetime.timedelta(hours=2)
            max_time = test_data['entry_time'].max() + datetime.timedelta(hours=2)

            sql_price = """ SELECT reporting_date, price_eur, price_usd, price_btc
                            FROM crypto_main.currency_hist_per_5_minute
                            where currency_id = '{}'
                            and reporting_date between '{}' and '{}'
                            order by reporting_date asc;
            """.format(self.coin_dict[coin][1], min_time, max_time)

            price_df = pd.read_sql(sql=sql_price, con=con_2)
            time_df = test_data['entry_time'] + self.lag
            price_df_temp = price_df.iloc[price_df['reporting_date'].searchsorted(time_df)]
            price_df_temp['change'] = price_df['price_eur'].diff().fillna(1)
            price_df_temp['change_bool'] = np.sign(price_df_temp['change'])
            prices_coins.append(price_df_temp['price_eur'])

            predicted_change = self.trained_models[coin].predict(test_features)
            data = {'entry_time' : test_data['entry_time'].values, coin: predicted_change,'price_eur_{}'.format(coin): price_df_temp['price_eur'].values}
            data = pd.DataFrame(data)
            data = data.set_index('entry_time')
            predictions.append(data)

        complete_pred = pd.concat(predictions, axis=1).fillna(0)

        total_value = []
        dict_coins =  defaultdict(float)
        dict_values = defaultdict(float)
        dict_coins_short = defaultdict(float)
        for row in complete_pred.itertuples():
            for i, (coin, _) in enumerate(self.coins_ratios):
                if getattr(row, coin) > 0:
                    if coin in dict_coins:
                        dict_money[coin] -= dict_coins_short[coin] * getattr(row, 'price_eur_{}'.format(coin))
                        dict_coins_short[coin] = 0
                        dict_money[coin] -= (dict_money[coin] / 10)
                        dict_coins[coin] += (dict_money[coin]/10) / getattr(row, 'price_eur_{}'.format(coin))
                        dict_values[coin] = dict_coins[coin]* getattr(row, 'price_eur_{}'.format(coin))
                    else:
                        dict_money[coin] = (dict_money[coin]*(9/10))
                        dict_coins[coin] = (dict_money[coin]/10) / getattr(row, 'price_eur_{}'.format(coin))
                        dict_values[coin] = dict_coins[coin] * getattr(row, 'price_eur_{}'.format(coin))
                elif getattr(row, coin) < 0:
                    if coin in dict_coins:
                        dict_money[coin] -= dict_coins_short[coin] * getattr(row, 'price_eur_{}'.format(coin))
                        dict_coins_short[coin] =  (dict_money[coin]/10) / getattr(row, 'price_eur_{}'.format(coin))
                        dict_money[coin] += (dict_coins[coin] + dict_coins_short[coin]) * getattr(row, 'price_eur_{}'.format(coin))
                        dict_coins[coin] = 0
                        dict_values[coin] = dict_coins[coin] * getattr(row, 'price_eur_{}'.format(coin))

            value_eur = sum(dict_money.values())
            value_coins = sum(dict_values.values())
            total_value.append(value_eur + value_coins)

        if plot:
            fig, ax1 = plt.subplots()
            fig.suptitle('Money and sentiment vs time according to sentiment trading', fontsize=14)

            ax1.plot(complete_pred.index, total_value, color='C1')
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            # set formatter
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            # set font and rotation for date tick labels
            plt.gcf().autofmt_xdate()

            ax1.set_xlabel("time")
            ax1.set_ylabel("Eur", color='C1')
            ax1.tick_params(axis='y', labelcolor='C1')

            #ax2 = ax1.twinx()
            #ax2.set_ylabel("Sentiment", color='C2')
            #ax2.plot(complete_pred.index, test_features['average_score'], color='C2')
            #ax2.tick_params(axis='y', labelcolor='C2')

            fig.tight_layout()

            return fig
            #for i in range(len(prices_coins)):
                #plt.plot(prices_coins[i])

if __name__ == '__main__':
    crypto_not = AlgorithmTrading(['XRP', 'LTC'], [0.2,0.8], 10000, 'Naive Bayes Gauss', 60)

    list_queries = ['total_tweets, change_total_tweets, entry_time, symbol',
                    'average_score, change_average_score, entry_time, symbol',
                    'total_tweets, change_total_tweets, entry_time, average_score, change_average_score, symbol',
                    'total_tweets, change_total_tweets, entry_time, symbol, retweets, followers',
                    'average_score, change_average_score, entry_time, symbol, retweets, followers',
                    'count_pos_tweets, count_neg_tweets, change_pos_tweets, change_neg_tweets, entry_time, symbol',
                    'count_pos_tweets, count_neg_tweets, change_pos_tweets, change_neg_tweets, followers, retweets, entry_time, symbol']

    for query in list_queries:
        print("Now looking at {}".format(query))
        crypto_not.trading_train(query=query)
        crypto_not.trading_test(query=query)
        crypto_not.perform_CV(query=query)
        crypto_not.plot_confusion_matrix(query=query)
