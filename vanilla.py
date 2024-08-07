#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:25:59 2024

@author: gustav
"""

import pickle
import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from lib.custom_openturns import log_pdf_x, \
    mu_post_truncated_normal, var_post_truncated_normal
import seaborn as sns

#%%

#Load meta models and data
with open('metamodels_data.p','rb') as f:
    
    metamodels_data = pickle.load(f)  
    
    metamodels = metamodels_data['metamodels']
    fgr_data = metamodels_data['data']
    

#Defaulting gb_sweeping and athermal release
metamodels = [
    ot.ParametricFunction(metamodel,[2,4],np.ones(2))
    for metamodel in metamodels
    ]

metamodels = metamodels_data['metamodels']
    
#%%

#Get measurement array
meas_v = fgr_data.meas_v

meas_unc = lambda v: ((v*0.05)**2 + 0.01**2) **.5

#Defining some likelihoods
likes = [ot.Normal(v,meas_unc(v)) for v in meas_v]

#%%

def logp(x):
    
    logp = 0
    
    for i,model in enumerate(metamodels):
        
        pred = model(x)
        
        logp += likes[i].computeLogPDF(pred)
        
    return [logp]

logp_ot = ot.PythonFunction(5,1,logp)

lbs = [0.15,0.1,0.1,0,.1]
ubs = [40,10,1,1,10]

initial_state = [20.,5.,.5,.5,1.]

support = ot.Interval(
    lbs,
    ubs
    )

samplers = [ot.RandomWalkMetropolisHastings(
    logp_ot,
    support,
    initial_state,
    ot.Uniform(-1,1),
    [i]
    ) for i in range(4)]

sampler = ot.Gibbs(samplers)

n = 2000

samples = sampler.getSample(n)

#%%
corner(samples.asDataFrame().iloc[500:,:4])