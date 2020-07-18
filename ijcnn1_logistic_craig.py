# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:20:28 2020

@author: vinusankars
"""

import numpy as np
from sampling_algos import sample_class

class logistic_regression():
    
    def __init__(self, size):        
        self.size = size
        self.w = np.zeros(size) + 0.01
    
    def predict(self, x):
        return 1/(1 + np.exp(np.dot(x, self.w)))
    
    def train_step(self, x, y, lr):
        
        wxy = np.dot(x, self.w)*y
        # f = np.log(np.prod(1 + np.exp(-wxy))) + 0.5*len(x)*10**-5*np.dot(self.w, self.w)
        del_f = np.zeros(len(x))
        
        for i in range(len(x)):
            del_f[i] = 1/(np.exp(wxy[i]) + 1)*y[i]*x[i] + 10**-5*self.w
        
        self.w += lr*del_f
        
