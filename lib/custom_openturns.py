#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:55:05 2024

@author: robertgc
"""

import openturns as ot
import numpy as np

class mu_post_truncated_normal(ot.OpenTURNSPythonFunction):
    
    def __init__(self,dim=0,ndim=3,lb=-100,ub=100,nexp=31):
        
        #Get dimension and total number of dimensions
        self._dim = dim
        self._ndim = ndim
        
        #Get lower and upper bound 
        self._lb = lb
        self._ub = ub
        
        self._nexp = nexp

        state_length = 2*ndim+nexp*ndim
        
        self._xindices = range(state_length)[2*ndim:][dim::ndim]
        
        self._state_length = state_length
        
        #Call superclass constructor
        super().__init__(state_length,4)
        
    def _exec(self,state):
        
        #Get dimsion and total number of dimensions
        dim = self._dim
        ndim = self._ndim
        
        nexp = self._nexp
        
        #Get upper and lower bounds 
        lb = self._lb
        ub = self._ub
        
        xindices = self._xindices
        
        #Get the xvalues
        x = np.array(
            [state[i] for i in xindices]
            )
                
        #Get sigma value
        sig = state[ndim+dim] ** .5
        
        #Calculate the hyperparameters of a truncated normal
        return [np.sum(x)/nexp, sig/np.sqrt(nexp),lb,ub]
        
        
class var_post_truncated_normal(ot.OpenTURNSPythonFunction):
    
    def __init__(self,dim=0,ndim=3,lb=1e-4,ub=100,nexp=31):
        
        #Set dimension and number of dimensions
        self._dim = dim
        self._ndim = ndim
        
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
        
        #Get dimsion and number of dimensions
        dim = self._dim
        nexp = self._nexp
        
        #Get lower and upper bounds 
        lb = self._lb
        ub = self._ub
        
        xindices = self._xindices

        #Get xvalues from the state
        x = np.array(
            [state[i] for i in xindices]
            )
        
        #Get mu
        mu = state[dim]
                
        #Calculate the hyperparameters of a truncated inverse gamma 
        return [2/np.sum((x-mu)**2), nexp/2, lb, ub]


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
        sig = state[dim+ndim] **.5
        
        #Use the metamodel to predict experiment 
        pred = metamodel(x)
        
        #Set logp
        logp = 0
        
        #Calculate and add likelihood
        logp += like.computeLogPDF(pred)
        
        #Add the logp coming from the hierachical part
        logp += ot.Normal(mu,sig).computeLogPDF(xi)
                        
        return [logp]