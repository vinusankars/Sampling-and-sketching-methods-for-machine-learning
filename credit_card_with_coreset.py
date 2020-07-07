# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:20:47 2020

@author: vinusankars
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sampling_algos import sample_class
from time import time

np.random.seed(42)

# read credit card data
data = np.array(pd.read_excel('../credit_card.xls', index_col=0))
attributes = data[0]
x = data[1:, :-1].astype('float32')
y = data[1:, -1].astype('float32')

start = time()
sample_method = sample_class()
_, u = sample_method.svm_coreset(x=x, s=1000, eps=0.2, delta=0.2, force=True)
importance_time = time()-start

def get_important_samples(u, s):
    
    temp = np.sort(u)
    cutoff = temp[-s]
    inds = (u >= cutoff)
    return inds

# generate plots
svm_coreset_time = []
random_sample_time = []
svm_coreset_score = []
random_sample_score = []

# get SVM full data timings
start = time()
clf = svm.SVC(gamma=1)
clf.fit(x, y)
svm_time = time()-start
svm_score = clf.score(x, y)

for i in range(10000, 30001, 2000):
    print("i = " + str(i), end='\r', flush=True)
    
    # svm_coreset
    start = time()
    inds = get_important_samples(sample_method.inds, i)
    clf_coreset = svm.SVC(gamma=1)
    clf_coreset.fit(x[inds], y[inds], sample_weight=u[inds])
    svm_coreset_time.append(time()-start)
    svm_coreset_score.append(clf_coreset.score(x, y))
    
    # random_coreset
    start = time()
    inds = np.random.randint(0, len(x))
    clf_random = svm.SVC(gamma=1)
    clf_random.fit(x[inds], y[inds])
    random_sample_time.append(time()-start)
    random_sample_score.append(clf_random.score(x, y))