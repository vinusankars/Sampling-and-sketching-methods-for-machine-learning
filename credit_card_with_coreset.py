# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:20:47 2020

@author: vinusankars
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sampling_algos import sample_class

# read credit card data
data = np.array(pd.read_excel('../credit_card.xls', index_col=0))
attributes = data[0]
x = data[1:, :-1].astype('float32')
y = data[1:, -1].astype('float32')

sample_method = sample_class()
S, u = sample_method.svm_coreset(x=x, s=1000, eps=0.2, delta=0.2, force=True)

clf = svm.SVC(gamma=1)
clf.fit(S, y[sample_method.inds], sample_weight=u)