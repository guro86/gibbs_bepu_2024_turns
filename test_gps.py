#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:36:08 2024

@author: robertgc
"""

#Import 
import pickle
import openturns as ot
import numpy as np
import matplotlib.pyplot as plt

#%%

#Load meta models 
with open('metamodels_data.p','rb') as f:

    metamodels_data =  pickle.load(f)  
    
    metamodels = metamodels_data['metamodels']
    fgr_data = metamodels_data['data']
    
#%%


def predict(x):
    
    preds = ot.Point()
    
    for metamodel in metamodels:
        preds.add(metamodel(x))
            
    return preds
   
predict_ot = ot.PythonFunction(5,31,predict)  

inp = [1,1,1,0.5,1.0]

pred = predict_ot(inp)

plt.plot(
    fgr_data.meas_v,
    pred,
    'o'
    )

l = np.linspace(0,.4)

plt.plot(l,l,'--')
plt.show()

#%%

Xtest = fgr_data.Xtest.values
ytest = fgr_data.ytest.values

#Loop over all experiments 
for exp in range(31):

    #plott test vs gp-predictions
    plt.plot(
        ytest[:,exp],
        metamodels[exp](Xtest).asDataFrame().values,
        'o'
        )
    
#Plot equalittyline
l = np.linspace(0,.4)
plt.plot(l,l,'--')

#and some labels
plt.xlabel('Test predictions [-]')
plt.ylabel('GP predictions [-]')

plt.show()