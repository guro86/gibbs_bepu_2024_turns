#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:23:06 2023

@author: gustav
"""

import data
from lib.gp import gp_ensemble
import matplotlib.pyplot as plt
import pickle

#%%

train_test_split_kwargs = {'random_state':19885}

fgr_data = data.fgr.dakota_data(train_test_split_kwargs=train_test_split_kwargs)
fgr_data.process()

Xtrain = fgr_data.Xtrain.values
ytrain = fgr_data.ytrain.values


gp = gp_ensemble(
       Xtrain = Xtrain,
       ytrain = ytrain,
       # use_cv_alpha = True
       )

gp.fit()

Xtest = fgr_data.Xtest.values
ytest = fgr_data.ytest.values

ypred = gp.predict_fast(Xtest)

plt.plot(
    ytest,
    ypred,
    'o'
    )


#%%
with open('gp_fgr.p','wb') as file:
    pickle.dump([fgr_data,gp], file)