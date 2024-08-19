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

initial_state = [9.78478,7.18681,0.292831,28.8707,0.518241,0.0262414,11.702,7.15259,0.504642,8.93815,7.20632,0.227912,10.7733,7.16673,0.423543,9.86638,7.16642,0.34344,9.52776,7.1736,0.19822,11.1469,7.12211,0.418397,9.08546,7.19024,0.14975,8.78541,7.22476,0.248703,10.3239,7.18289,0.342643,7.50655,7.23985,0.221591,9.47391,7.21092,0.235646,8.47214,7.22322,0.198574,9.8565,7.18337,0.308543,10.2439,7.17974,0.321506,8.07999,7.18148,0.280854,8.72788,7.20715,0.201784,11.8741,7.1397,0.468259,7.44071,7.2692,0.183411,9.93625,7.1716,0.308075,10.22,7.18294,0.343424,9.60504,7.19412,0.290279,16.2637,7.10373,0.572483,8.60862,7.20276,0.248138,8.75301,7.17151,0.241266,9.777,7.1634,0.317432,9.342,7.24039,0.116625,9.04912,7.17853,0.0566619,10.0503,7.19752,0.322601,15.0076,7.13199,0.536823,5.12208,7.224,0.141905,9.88428,7.19038,0.30269]#99


lbs_var = np.array([0.1,0.1,0.1])**2
ubs_var = np.array([40,20,10])**2

support = ot.Interval(
    lbs + lbs_var.tolist() + nexp * lbs,
    ubs + ubs_var.tolist() + nexp * ubs
    )
    
#%%
# Remove restriction on the proposal probability of the origin
ot.ResourceMap.SetAsScalar("Distribution-QMin", 0.0)
ot.ResourceMap.SetAsScalar("Distribution-QMax", 1.0)

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
    logp = ot.Function(
    log_pdf_x(
        exp=exp,
        like=likes[exp],
        metamodel=metamodels[exp],
        ndim = ndim
        )
    )

    base_index = 2*ndim+ndim*exp

    samplers += [
        ot.RandomWalkMetropolisHastings(
            logp,
            support,
            initial_state,
            ot.Normal([0.0] * 3, [20.0, 5.0, 0.5]),
            [base_index + i for i in range(ndim)]
            )
        ]
#%%

sampler = ot.Gibbs(samplers)

#%%

samples = sampler.getSample(24000)

acceptance = [sampler.getMetropolisHastingsCollection()[i].getAcceptanceRate() for i in range(len(samplers))]
print(acceptance)

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