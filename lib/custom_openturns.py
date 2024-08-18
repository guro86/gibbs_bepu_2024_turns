#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:55:05 2024

@author: robertgc
"""

import openturns as ot
import numpy as np
from time import time

class mu_post_truncated_normal(ot.OpenTURNSPythonFunction):
    
    def __init__(self,dim=0,ndim=3,lb=-100,ub=100,nexp=31):
        
        #Get dimension and total number of dimensions
        self._dim = dim
        self._ndim = ndim
        
        #Get lower and upper bound 
        self._lb = lb
        self._ub = ub
        
        # Total number of experiments
        self._nexp = nexp

        state_length = 2*ndim+nexp*ndim
        
        self._xindices = range(state_length)[2*ndim:][dim::ndim]
        
        self._state_length = state_length
        
        #Call superclass constructor
        super().__init__(state_length,4)
        
    def _exec(self,state):
        
        # posterior mean of mu = empirical mean of the x values
        post_mean = np.mean([state[i] for i in self._xindices])
        
        # posterior std of mu = prior sigma / sqrt(nexp)
        post_std = np.sqrt(state[self._ndim + self._dim] / self._nexp)
        
        # Hyperparameters of a truncated normal
        return [post_mean, post_std, self._lb, self._ub]
        
        
class var_post_truncated_normal(ot.OpenTURNSPythonFunction):
    
    def __init__(self,dim=0,ndim=3,lb=1e-4,ub=100,nexp=31):
        
        #Set dimension and number of dimensions
        self._dim = dim
        self._ndim = ndim
        
        # Total number of experiments
        self._nexp = nexp
        
        #Set lower and upper bounds 
        self._lb = lb
        self._ub = ub
        
        state_length = 2*ndim+nexp*ndim
        
        self._state_length = state_length
        
        self._xindices = range(state_length)[2*ndim:][dim::ndim]
       
        #Call super class constructor
        super().__init__(state_length,4)
        
    def _exec(self,state):

        #Get mu
        mu = state[self._dim]
        
        # Get squares of centered xvalues from the state
        squares = [(state[i] - mu)**2 for i in self._xindices]

        post_lambda = 2.0 / np.sum(squares) # rate lambda =  1 / beta
        post_k = self._nexp / 2.0 # shape
        
        # Hyperparameters of a truncated inverse gamma
        return [post_lambda, post_k, self._lb, self._ub]


#Create a logpdf for the latent parameters
class log_pdf_x(ot.OpenTURNSPythonFunction):
    
    def __init__(self, dim, exp, like, metamodel, ndim=3, nexp=31):
    
        self._ndim = ndim
        self._dim = dim
        self._exp = exp
        self._like = like
        self._metamodel = metamodel
        
        state_length = 2*ndim+nexp*ndim
        
        self._xindices = range(state_length)[2*ndim:][exp*ndim:ndim+exp*ndim]
        
        #Perhaps improve?!
        super().__init__(state_length,1)
        
    def _exec(self, state):
        
        #Get dimension and experiment number 
        dim = self._dim
        # exp = self._exp
        
        xindices = self._xindices
        
        #Get number of total dimensions
        ndim = self._ndim
        
        #Get the metamodel for the experiment
        metamodel = self._metamodel
        
        #get likelihood
        like = self._like
        
        #Get the x indices of the experiment
        x = np.array(
            [state[i] for i in xindices]
            )
        
        #get xindex for dimension (within experiment)
        xi = x[dim]
        
        #Get mu
        mu = state[dim]
        
        #get sig
        sig = np.sqrt(state[dim+ndim])
        
        #Use the metamodel to predict experiment 
        pred = metamodel(x)

        #Calculate loglikelihood
        logp = like.computeLogPDF(pred)
        
        #Add the logp coming from the hierarchical part
        #logp += ot.Normal(mu,sig).computeLogPDF(xi) # slow
        logp += ot.DistFunc.logdNormal((xi - mu) / sig) - np.log(sig) # 6 x faster
        # The - np.log(sig) term could be removed as it is constant w.r.t. x
                        
        return [logp]
