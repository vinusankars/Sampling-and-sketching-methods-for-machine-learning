# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:20:47 2020

@author: vinusankars
"""

import pandas as pd
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sampling_algos import sample_class
from time import time

np.random.seed(42)

# read credit card data
data = np.array(pd.read_excel('../credit_card.xls', index_col=0))
attributes = data[0]
x = data[1:, :-1].astype('float32')
y = data[1:, -1].astype('int')

# generate plots
svm_coreset_time = []
random_sample_time = []
svm_coreset_score = []
random_sample_score = []
size = []

sample_method = sample_class()
_ = sample_method.svm_coreset(x=x, s=1000)
    
# get SVM full data timings
start = time()
clf = svm.SVC(gamma=1)
clf.fit(sample_method.x, y)
svm_time = time()-start 
svm_score = clf.score(sample_method.x, y) 

epses = [0.1+j*0.01 for j in range(11)][::-1]
for i in epses:
    print("i = " + str(i))
    
    # svm_coreset
    start = time()
    sample_method = sample_class()
    _, u = sample_method.svm_coreset(x=x, s=1000, eps=i, delta=i, force=True)
    inds = sample_method.inds
    clf_coreset = svm.SVC(gamma=1)
    clf_coreset.fit(sample_method.x[inds], y[inds], sample_weight=u[inds])
    svm_coreset_time.append(time()-start)
    svm_coreset_score.append(clf_coreset.score(sample_method.x, y))
    
    # random_coreset
    start = time()
    inds = np.random.randint(0, len(x), int(sample_method.m))
    clf_random = svm.SVC(gamma=1)
    clf_random.fit(sample_method.x[inds], y[inds])
    random_sample_time.append(time()-start)
    random_sample_score.append(clf_random.score(sample_method.x, y))
    
    size.append(int(sample_method.m))
    
plt.figure(figsize=(10, 5))
plt.plot(size, [svm_score]*11, 'r--', label='All data SVM')
plt.plot(size, svm_coreset_score, 'b-o', label='SVM coreset')
plt.plot(size, random_sample_score, 'g-o', label='Uniform randomly sampled SVM')
plt.xlabel('Coreset size')
plt.ylabel('Training score')
plt.title('SVM coresets on credit card dataset')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(size, [svm_time]*11, 'r--', label='All data SVM')
plt.plot(size, svm_coreset_time, 'b-o', label='SVM coreset')
plt.plot(size, random_sample_time, 'g-o', label='Uniform randomly sampled SVM')
plt.xlabel('Coreset size')
plt.ylabel('Training time')
plt.title('SVM coresets on credit card dataset')
plt.legend()
plt.show()