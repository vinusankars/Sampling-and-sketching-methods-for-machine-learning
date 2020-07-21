# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:20:28 2020

@author: vinusankars
"""

import numpy as np
from matplotlib import pyplot as plt
from time import time
#from sklearn.linear_model import LogisticRegression
from sampling_algos import sample_class

np.random.seed(42)

class LogisticRegression():
    
    def __init__(self):
        self.w = np.zeros(54)
        self.b = 0
        
    def predict(self, x):
        return np.stack([1/(1+np.exp(-(self.w*x[i]).sum())) for i in range(len(x))])
    
    def fit(self, x, y, weight, lr=0.001):
        
        for itr in range(100):
            y_ = self.predict(x)
            del_b = -2*np.sum((y-y_)*(1-y_)*y_)
            sample = (y-y_)*(1-y_)*y_
            del_w = -2*np.stack([sample[i]*weight[i]*x[i] + 10**-5*self.w for i in range(len(sample))]).sum(0)
            self.b -= lr*del_b
            self.w -= lr*del_w
            
    def score(self, x, y):
        
        y_ = self.predict(x)
        ind = (y_ >= 0.5)
        y_ = y_*0 - 1
        y_[ind] = 1
        return (y_ == y).sum()/len(y)

# read covtype data
with open('../covtype.binary', 'r') as f:
    data = np.stack(f.read().split('\n'))[: -1]
    np.random.shuffle(data)
    data = data[: 200]
    
X = np.zeros((len(data), 54))
y = np.zeros(len(data))

for i in range(len(data)):
    tmp = data[i].split()
    y[i] = float(tmp[0])*2-3
    
    for j in tmp[1: ]:
        X[i][int(j.split(':')[0])-1] = float(j.split(':')[1])
        
X = X - np.mean(X, 0)
X = X/(np.std(X, 0)+0.1)

print("\nTraining regressor...")

start = time()
reg  = LogisticRegression()
reg.fit(X, y, np.ones(len(X)))
print("Time for LogR", time()-start)
print("Score for LogR", reg.score(X, y))

print("\nRunning CRAIG...")    
frac = 0.9

start = time()
sc = sample_class()
pos_class = (y == 1)
neg_class = (y == -1)
sigma_neg, gammas_neg = sc.craig(X[neg_class], frac)
sigma_pos, gammas_pos = sc.craig(X[pos_class], frac)
#sigma = [sigma_pos[int(i/2)]*int(i%2==0) + sigma_neg[int((i+1)/2)]*int(i%2==1) for i in range(len(sigma_neg) + len(sigma_pos))]
#gammas = [gammas_pos[int(i/2)]*int(i%2==0) + gammas_neg[int((i+1)/2)]*int(i%2==1) for i in range(len(sigma_neg) + len(sigma_pos))]
sigma = sigma_pos.tolist() + sigma_neg.tolist()
sigma = np.stack(sigma)
#np.random.shuffle(sigma)
gammas = gammas_pos.tolist() + gammas_neg.tolist()
gammas = np.stack(gammas)

print("\nTraining regressor with CRAIG...")
reg_craig  = LogisticRegression()
reg_craig.fit(X[sigma], y[sigma], gammas)
print("Time for LogR with CRAIG", time()-start)
print("Score for LogR with CRAIG", reg_craig.score(X, y))

start = time()
sigma = np.random.randint(0, len(X), int(frac*len(X)))
gammas = np.ones(len(sigma))*(len(X)/len(sigma))

print("\nTraining regressor with random coresets...")
reg_craig  = LogisticRegression()
reg_craig.fit(X[sigma], y[sigma], gammas)
print("Time for LogR with random coresets", time()-start)
print("Score for LogR with random coresets", reg_craig.score(X, y))