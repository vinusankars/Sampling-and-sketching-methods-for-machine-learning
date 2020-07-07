# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:57:38 2020

@author: vinusankars
"""

import numpy as np

class sample_class():
    
    def __init__(self):
        pass
    
    def svm_coreset(self, x, s, eps=0.1, delta=0.1, force=True):
        """
        Training Support Vector Machines using Coresets
        Baykal et al. [ArXiv]
        
        Output:
        eps-coreset (S,u) with probability 1-delta for query space ~F
        
        If force == True, s >= m will be forced.
        """
        
        assert 0 <= eps <= 1, "Epsilon range error"
        assert 0 <= delta <= 1, "Delta range error"
        assert len(x.shape) <= 2, "Input data shape error"
        
        self.x = x.astype('float32')
        self.eps = eps
        self.delta = delta
        self.n = len(self.x)
        self.s = s # size of output
        
        # Preprocessing, satisfying assumptions 4 and 5
        if len(self.x.shape) == 1: # making x a 2D vector
            self.x = self.x.reshape((-1, 1))
            
        self.x -= np.mean(self.x, 0)
        self.x /= np.max(np.std(self.x, 0))
        self.d = self.x.shape[1]
        
        gamma = np.zeros(self.n)        
        for i in range(self.n):            
            gamma[i] = (1 + np.log10(self.n) + np.std(self.x[i])*np.log10(self.n)**2)/self.n
        
        t = np.sum(gamma)
        m = t/self.eps**2 * (self.d*np.log10(t) + t*np.log10(1/self.delta))
        self.m = m
        self.gamma = gamma
        self.t = t
           
        if force == True:        
            self.s = max(self.s, self.m)
            
        K = np.random.multinomial(self.s, gamma/t)
        inds = (K > 0)
        S = self.x[inds]
        u = np.zeros(self.n)
        
        for i in range(self.n):
            u[i] = t*K[i]/(gamma[i]*len(S))
            
        self.S = S
        self.u = u[inds]
        self.inds = inds
        
        return (self.S, self.u)