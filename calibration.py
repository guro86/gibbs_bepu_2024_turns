#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:35:23 2024

@author: robertgc
"""
# %%
import pickle
import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from lib.custom_openturns import log_pdf_x, \
    mu_post_truncated_normal, var_post_truncated_normal
import pandas as pd
import scipy.stats as stats

#%%

#Load meta models and data
with open('metamodels_data.p','rb') as f:
    
    metamodels_data = pickle.load(f)  
    
    metamodels = metamodels_data['metamodels']
    fgr_data = metamodels_data['data']
    

#Defaulting gb_sweeping and athermal release
metamodels = [
    ot.ParametricFunction(metamodel,[2,4],[0.99,1.0])
    for metamodel in metamodels
    ]
    
#%%

#Get measurement array
meas_v = fgr_data.meas_v

meas_unc = lambda v: ((v*0.05)**2 + 0.01**2) **.5

#Defining some likelihoods
likes = [ot.Normal(v,meas_unc(v)) for v in meas_v]

#%%

#Random vector for sampling of mu
mu_rv = ot.RandomVector(
    ot.TruncatedNormal()
    )

#Random vector for sampling of sigma
var_rv = ot.RandomVector(
    ot.TruncatedDistribution(
        ot.InverseGamma(),
        0.0,
        1.0
        )
    )


#Functions for caluclating hyper parameters given state
mu_post = mu_post_truncated_normal()
var_post = var_post_truncated_normal()


#%%

ndim = 3
nexp = 31

lbs = [0.1,0.1,1e-4]
ubs = [40,10,1]

means = np.array([20.,5.,.5])

initial_state = np.ones(2*ndim+ndim*nexp)

initial_state[:ndim] = means
# use different initial sigmas to distinguish logpdfs
initial_state[ndim:2*ndim] = np.array([20.0, 15.0, 0.5])**2

initial_state[2*ndim::3] = 20.
initial_state[2*ndim+1::3] =  5. 
initial_state[2*ndim+2::3] = .5 

lbs_var = np.array([0.1,0.1,0.1])**2
ubs_var = np.array([40,20,10])**2

support = ot.Interval(
    lbs + lbs_var.tolist() + nexp * lbs,
    ubs + ubs_var.tolist() + nexp * ubs
    )
    
#%%
samplers = []
samplers = [
    ot.RandomVectorMetropolisHastings(
        mu_rv,
        initial_state,
        [i],
        ot.Function(
            mu_post_truncated_normal(dim=i,ndim=3,lb=lbs[i],ub=ubs[i],nexp=nexp)
            )
        )
    for i in range(ndim)
      ]


samplers += [
    ot.RandomVectorMetropolisHastings(
        var_rv,
        initial_state,
        [ndim+i],
        ot.Function(
            var_post_truncated_normal(dim=i,ndim=3,lb=lbs_var[i],ub=ubs_var[i],nexp=nexp)
            )
        )
    for i in range(ndim)
    ]

for exp in range(nexp): 
    for dim in range(ndim):
        
        logp = ot.Function(
        log_pdf_x(
            dim=dim,
            exp=exp,
            like=likes[exp],
            metamodel=metamodels[exp],
            ndim = 3
            )
        )
        
        i = 2*ndim+ndim*exp+dim
        
        samplers += [
            ot.RandomWalkMetropolisHastings(
                logp,
                support,
                initial_state,
                ot.Uniform(-1,1),
                [i]
                )
            ]
#%%

sampler = ot.Gibbs(samplers)

#%%

samples = sampler.getSample(24000)

#%%

names = ['diff','gbsat','crack']

hypost = samples.asDataFrame().iloc[:,:6] # interesting to look at whole sample
hypost.iloc[:,-3:] = hypost.iloc[:,-3:].apply(np.sqrt)


hypost.columns = [f'{p}_{{{n}}}' for p in ['$\mu$','$\sigma$'] for n in names]

#%%

corner(hypost)
plt.show()

#%%

mu = hypost.iloc[:,:3]
sig = hypost.iloc[:,3:6]

mu.columns = np.arange(3)
sig.columns = np.arange(3)

a = (lbs-mu)/sig
b = (ubs-mu)/sig

marg_samples = stats.truncnorm(loc=mu,scale=sig,a=a,b=b).rvs(mu.shape)

marg_samples_pd = pd.DataFrame(marg_samples)
marg_samples_pd.columns = ['diff','gbsat','crack']

corner(marg_samples_pd)
plt.show()

#%%

mean_pred = np.array([metamodels[i](marg_samples).computeMean()[0] for i in range(31)])
ub_pred = np.array([metamodels[i](marg_samples).computeQuantile(0.95)[0] for i in range(31)])
lb_pred = np.array([metamodels[i](marg_samples).computeQuantile(0.05)[0] for i in range(31)])

yerr = np.abs(np.column_stack([lb_pred,ub_pred]).T - mean_pred)


l = np.linspace(0 , .5)

plt.errorbar(meas_v, mean_pred,yerr,fmt='o')
# plt.plot(meas_v,mean_pred,'o')
# plt.plot(meas_v,ub_pred,'o')
# plt.plot(meas_v,lb_pred,'o')

plt.xlabel('Measured fgr [-]')
plt.ylabel('GP predicted fgr [-]')


plt.plot(l,l,'--')
plt.show()