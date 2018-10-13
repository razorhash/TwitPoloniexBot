# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:02:49 2018

@author: SebastiaanHersmisADC
"""
import pandas as pd

class Orders(object):
    """Orders contains the structure of the generated orders by the 
    different algorithms. """
    
    def __init__(self, orders):
        
        # check if this is a pandas DataFrame
        # https://stackoverflow.com/questions/14808945/
        if not isinstance(orders, pd.DataFrame):
            raise NotImplementedError("Orders is not the correct format")
        
        self.orders = orders
    
    def get_orders(self):
        return self.orders