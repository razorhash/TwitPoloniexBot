    
        
         
        # a sell or buy can influence subsequent positions, so calculate iteratively
        for observation in range(1, len(data.index)):
            
            date = data.index[observation]
            
            # investment this period is zero
            investment_capital_period = 0
            
            # amount of currency_ids initially same as last period
            for currency_id in np.unique(self.orders.currency_id):
                data[currency_id + '_position'].iloc[observation] = data[currency_id + '_position'].iloc[observation-1]                    
          
            # at each point, compute size of each position (cash and currencies), and record actions
            if(data.index[observation] in self.orders.index):
                                
                action_df = pd.DataFrame(columns=list(["Currency","NominalAmount", "CapitalAmount"]))
                
                # could be multiple actions
                for index, action in self.orders.loc[date].iterrows():    
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
    
    
    
    ####
    
 ## ---- Perform strategy ----------------------------------------------
        # moving average of 1 day = 288 obs, 6 hours = 72 obs
        SMA_fill = np.full(self.MA_window-1, np.nan)
        SMA = np.convolve(self.data[self.pair_id].as_matrix(), np.ones((self.MA_window,))/self.MA_window, mode='valid')
        SMA = np.concatenate((SMA_fill, SMA), axis=0)
        
        # upper and lower bound
        std_upper = SMA + self.band_with * np.std(self.data[self.pair_id].as_matrix())
        std_lower = SMA - self.band_with * np.std(self.data[self.pair_id].as_matrix())
        
        # see where we breach the boundary
        sell = -1 * (std_upper < self.data[self.pair_id])
        # only include first breach before return
        sell = np.append(0, np.diff(sell))
        sell[sell > 0] = 0
        
        # see where we breach the boundary        
        buy = 1 * (std_lower > self.data[self.pair_id])  
        # only include first breach before return
        buy = np.append(0, np.diff(buy))
        buy[buy < 0] = 0
            
        # initiate vector and convert to correct format
        signals = pd.DataFrame(index = self.data.index) 
        signals["signal"] = sell + buy      
        signals["currency_id"] = np.nan
        
        # filter to actions
        signals = signals[signals.signal != 0]          
        
        # buy means buy self.pair_id1 and sell self.pair_id2
        buy_pair = signals[signals.signal == 1]
        buy_signals = pd.DataFrame({"currency_id": self.pair_id1, "signal": 1}, index = buy_pair.index)
        buy_signals = buy_signals .append(pd.DataFrame({"currency_id": self.pair_id2, "signal": -1}, index = buy_pair.index))
        
        # sell
        sell_pair = signals[signals.signal == -1]
        sell_signals = pd.DataFrame({"currency_id": self.pair_id1, "signal": -1}, index = sell_pair.index)
        sell_signals = sell_signals .append(pd.DataFrame({"currency_id": self.pair_id2, "signal": 1}, index = sell_pair.index))
        
        # combine 
        signals = pd.concat([buy_signals, sell_signals])

        # return sell_pair
        signals = signals.sort_index()
        return signals    