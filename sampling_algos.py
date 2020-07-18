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
            
        self.x /= np.max(np.std(self.x, 0))
        self.x -= np.mean(self.x, 0)
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
        self.inds = inds
        self.u = u
        
        return (self.S, self.u)
    
    def craig(self, x, size=0.3):
        """
        Coresets for Data-efficient Training of Machine Learning Models
        Mirzasoleiman et al. [ICML 2020]
        
        Output: Coreset S with their per-element stepsizes gammas
        
        Assumptions:
        Eqn (9) holds.
        """
        
        sigma = []
        s0 = np.zeros(x.shape[-1]).tolist()
        self.x = x    
        self.size = len(x)*size
        gammas = np.zeros(self.size)
        
        for i in range(self.size):             
            e, L = np.inf, np.inf    
            S  = x[sigma].tolist()
            
            for j in range(len(x)):                
                if j not in sigma:
                    
                    s_ = np.stack(S + [x[j], s0])
                    l = 0                    
                    for k in range(len(x)):
                        l += min(np.std(s_ - x[k], 0))
                        
                    if l < L:
                        L = l
                        e = j
                        
            sigma.append(e)
        
        for i in range(len(x)):
            gammas[np.argmin(np.std(x[sigma] - x[i], 0))] += 1
                
        self.sigma = np.stack(sigma)
        self.gammas = gammas
        
        return self.sigma, self.gammas