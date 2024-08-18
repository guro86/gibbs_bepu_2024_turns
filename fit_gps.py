#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:47:23 2024

@author: robertgc
"""

import data
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import multiprocessing as mp
import pickle

#%%

#Training test split random state
train_test_split_kwargs = {'random_state':19885}

#A data object
fgr_data = data.fgr.dakota_data(
    train_test_split_kwargs=train_test_split_kwargs
    )

#Preprocessing of the data (read and split etc.)
fgr_data.process()

#Training values 
Xtrain = fgr_data.Xtrain.values
ytrain = fgr_data.ytrain.values

#Testing values
Xtest = fgr_data.Xtest.values
ytest = fgr_data.ytest.values

#Dimensions
dimension = 5

#Constant basis
basis = ot.ConstantBasisFactory(dimension).build()

#Squared exponential kernel with dimension lenbght scales
#Unity amplitude
covarianceModel = ot.SquaredExponential([1.0] * dimension, [1.0])
#We set a little bit of noise to regularize the fitting
covarianceModel.setNuggetFactor(1e-3)

#Function returning the GP for exp = i
def func(exp):

    #We do the algorithm 
    algo = ot.KrigingAlgorithm(
        Xtrain, 
        ytrain[:,exp].reshape(-1,1), 
        covarianceModel, 
        basis)

    #We run it 
    algo.run()
    
    #Get results 
    result = algo.getResult()
    
    #and the metamodel
    metamodel = result.getMetaModel()
    
    print(f'returning metamodel {exp}')
  
    return metamodel

with mp.Pool() as pool:
    metamodels = pool.map(func, range(31))

#%%

#Loop over all experiments 
for exp in range(31):

    #plott test vs gp-predictions
    plt.plot(
        ytest[:,exp],
        metamodels[exp](Xtest).asDataFrame(),
        'o'
        )
    
#Plot equalittyline
l = np.linspace(0,.4)
plt.plot(l,l,'--')

#and some labels
plt.xlabel('Test predictions [-]')
plt.ylabel('GP predictions [-]')


with open('metamodels_data.p','wb') as file:
    pickle.dump({'metamodels':metamodels,'data':fgr_data}, file)