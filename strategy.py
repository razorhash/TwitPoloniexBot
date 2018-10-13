# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:02:49 2018

@author: SebastiaanHersmisADC
"""
from abc import ABCMeta, abstractmethod

class Strategy(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of orders,
    which has the Orders class."""
    
    __metaclass__ = ABCMeta
     
    # child method of this strategy should have a generate_signals ()
    @abstractmethod
    def generate_orders(self):
        """An implementation is required to return the orders """
        raise NotImplementedError("Should implement generate_orders()!")