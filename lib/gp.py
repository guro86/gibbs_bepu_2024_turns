#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:20:27 2022

@author: robertgc
"""

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

class gp_ensemble():
    
    def __init__(self,**kwargs):
        
        #Get the instances from kwargs
        self.Xtrain = kwargs.get('Xtrain',None)
        self.ytrain = kwargs.get('ytrain',None)
        
        #Gps and dimensions
        self.gps = None 
        self.dims = None
                
        #Alha for the diagnoal of K during fitting 
        self.alpha = kwargs.get('alpha',1e-2)
        
        #Dist for alpha, used if cross validation is on
        self.alpha_dist = uniform(loc=1e-10,scale=1e-1 - 1e-10)
        
        #Number of jobs for validation
        self.n_jobs_alpha = kwargs.get('n_jobs_alpha',4)
        
        #Switch for cross validate alpha 
        self.use_cv_alpha = kwargs.get('use_cv_alpha',False)
        
        #amplitudes and length scales 
        self.amplitudes                 = None 
        self.length_scales              = None
        self.alphas                     = None 
        self.y_train_means              = None 
        self.y_train_stds               = None  
        self.Xtrain_scaled              = None  
        self.Xtrain_scaled_squared_sum  = None  
        
        #List for the fitted Gaussian-processes
        self.gps = []
        
    def _ks(self,X):
        
        #Get relevant quantities
        length_scales = self.length_scales
                
        #Get the Xtrain data scaled by the length_scales
        Xtrain_scaled = self.Xtrain_scaled
        
        #Get amplitudes
        amplitudes = self.amplitudes
        
        #Get the scaled data summed over the last axis
        Xtrain_scaled_squared_sum = self.Xtrain_scaled_squared_sum

        #Scale the X data
        X_scaled = (X / length_scales[:,None,:])
               
        #Square and sum
        # X_scaled_squared_sum = np.sum(X_scaled ** 2,axis=-1)
        X_scaled_squared_sum = (X_scaled ** 2).sum(axis=-1)
          
        #Calculate the dot-product between the scaled vectors
        scaled_dot = Xtrain_scaled @ X_scaled.transpose(0,2,1)
        
        #Calculate the squared distance 
        sqdist = X_scaled_squared_sum[:,None,:] + \
            Xtrain_scaled_squared_sum[:,:,None] - \
                2*scaled_dot
        
        #Put the square distance in exp and scale with amplitudes
        ks = np.exp(-.5 * sqdist) 
    
        #Scale with the amplitudes
        ks = ks * amplitudes[:,None,None]
        
        return ks
    
    def predict_fast(self,X,ks=None):
        
        alphas = self.alphas
        y_train_means = self.y_train_means
        y_train_stds = self.y_train_stds
        
        #Get ks and scaled with amplitudes
        if ks is None:
            ks = self._ks(X) 
        
        #Multiply with alphas
        pred = alphas[:,None,:] @ ks
        
        #Undo normalization
        pred = pred * y_train_stds[:,None,None] + y_train_means[:,None,None]
        
        #Squeeze (better way in the future)
        pred = np.squeeze(pred)
        
        #Transpose
        return pred.T
    
    def predict_der_fast(self,X,ks=None):
        

        #If ks is not provided, calc
        if ks is None:
            ks = self._ks(X)
        
        #Get quantities 
        Xtrain = self.Xtrain
        length_scales = self.length_scales
        alphas = self.alphas
        y_train_stds = self.y_train_stds
        
        #Calculate the diffs
        diff = Xtrain.T[:,:,None] - X.T[:,None,:]
        
        #Calculate differentiated ks
        ks2 = diff / length_scales[:,:,None,None]**2 * ks[:,None,:,:]
        
        #Calculate non-normalized derivatives
        der = alphas[:,None,None,:] @ ks2
        
        #Undo normalization
        der *= y_train_stds[:,None,None,None]
        
        #Return derivatives, remove dimension that is always one
        #Transpose so that we give exp x nX x dimensions        
        return der[:,:,0,:].transpose(0,2,1)


    def predict_pred_and_der_fast(self,X):
        
        #Get ks only ones 
        ks = self._ks(X)
        
        #Get predictions and derivatives based on X and ks
        pred = self.predict_fast(X,ks)
        der = self.predict_der_fast(X,ks)
        
        #Return predictions and derivatives
        return pred, der
    
    def predict(self,X,**kwargs):
        
        #Get return std kwarg
        return_std = kwargs.get('return_std',False)
        
        #Get gps
        gps = self.gps 
        
        #Stack predictions 
        pred = np.column_stack([gp.predict(X) for gp in gps])
        
        
        #If std is requested
        if return_std:
        
            #Stack std of all gps
            std = np.column_stack(
                [gp.predict(X,return_std=True)[-1] for gp in gps]
                )
            
            #Return pred and std
            return pred, std
        
        #If stds are requested, return pred only
        else:
            
            #Return pred
            return pred
    
    def predict_der(self,X):
        
        #Get all the individual gps
        gps = self.gps
        
        #Calculate and stack all derviatives 
        J = np.stack(
            [gp.predict_der(X).T for gp in gps]
            )
        
        #Return
        return J
    
    def predict_i(self,X,i):
        
        #Get the gp
        gp = self.gps[i]
        
        #Return the prediction of the ith gp
        return gp.predict(X)
                
    
    def sample_y(self,X):
        
        #Get the gps 
        gps = self.gps 
        
        #Loop and sample the gps
        samples = np.column_stack([gp.sample_y(X) for gp in gps])
        
        #Return samples 
        return samples
    
    def fit(self):
        
        #Get training data 
        Xtrain = self.Xtrain
        ytrain = self.ytrain
        
        use_cv_alpha = self.use_cv_alpha
        
        n_jobs_alpha = self.n_jobs_alpha
        
        #Calc dimensions of inpu
        dims = Xtrain.shape[-1]
        
        #Store dimensions 
        self.dims = dims
        
        alpha_dist = self.alpha_dist
        
        #A list of one gp per output dimension
        gps = [self._factory() for i in range(ytrain.shape[-1])]
       
       
        for i,gp in enumerate(gps): 
            
            ytrain_i = ytrain[:,i]
                
            
            if use_cv_alpha:
                
                print('Cross validating alpha for {}-th gp'.format(i))
                
                search = RandomizedSearchCV(
                    gp,
                    {'alpha':alpha_dist},
                    n_jobs = n_jobs_alpha
                    )
                
                search.fit(
                    Xtrain,
                    ytrain_i,
                    )
                
                gps[i] = search.best_estimator_
                
            else:
                
                gp.fit(
                    Xtrain,
                    ytrain_i
                    )                

        #Store trained gps
        self.gps = gps
        
        #Prepare
        self._prepare()
        
    def _prepare(self):
        
        #get the gps 
        gps = self.gps    
        
        #Get the Xtrain values
        Xtrain = self.Xtrain
        
        #get all length_scales 
        length_scales = np.stack([
            gp.kernel_.k2.get_params()['length_scale'] for gp in gps    
            ])
        
        
        #get amplitudes
        amplitudes = np.stack([
            gp.kernel_.k1.get_params()['constant_value'] for gp in gps    
            ])


        #get alphas
        alphas = np.stack([
              gp.alpha_ for gp in gps    
              ])
        
        
        #get mean of y_train
        y_train_means = np.stack(
            [
                gp._y_train_mean for gp in gps
                ]
            )
        
        #Get ytrain stds
        y_train_stds = np.stack(
            [
                gp._y_train_std for gp in gps
                ]
            )
        
        #Calculated Xtrain scaled and scaled square sums
        Xtrain_scaled = (Xtrain / length_scales[:,None,:])
        Xtrain_scaled_squared_sum = np.sum(Xtrain_scaled ** 2,axis=-1)
        
        #Store for future predictions
        self.length_scales = length_scales
        self.amplitudes = amplitudes
        self.alphas = alphas
        self.y_train_means = y_train_means
        self.y_train_stds = y_train_stds
        self.Xtrain_scaled = Xtrain_scaled
        self.Xtrain_scaled_squared_sum = Xtrain_scaled_squared_sum
        
    #Internal factory function to create a local gaussian process
    def _factory(self):
        
        #Get fimenstions 
        dims = self.dims 
        
        #Get alpha
        alpha = self.alpha
        
        #Create kernel
        kernel = 1 * RBF(
            length_scale=np.ones(dims),
            length_scale_bounds=(1e-10,1e10)
            )
                
        #Create gp
        gp = my_gp(
            kernel = kernel,
            normalize_y=True,
            alpha = alpha,
            )
        
        #return gp
        return gp

class my_gp(GPR):
    
    #Function that calculates derivative
    #(give an RBF kernel)
    def predict_der(self,X):
        
        #std used in normalization
        y_train_std = self._y_train_std
        
        #Get kernel 
        kernel = self.kernel_
        
        #Get alpha vector
        alpha = self.alpha_ 
        
        #Get training data
        Xtrain = self.X_train_

        #Get length scale/scales
        l = self.kernel_.get_params()['k2__length_scale']
        
        #Fix dimensions if np array
        #i.e. l in the first dimension and 
        #append the last two
        if isinstance(l,np.ndarray):
            l = l[:,np.newaxis,np.newaxis]
            
        #Calculate diffs dimension for dimension
        #All combinations of diffs are calculated 
        #for all different dimensions (first axis)
        diff = Xtrain[np.newaxis,:].transpose(2,1,0) - \
            X[np.newaxis,:].transpose(2,0,1)
        
        #Use kernel and diffs to evaluate ks
        ks = kernel(Xtrain,X) * diff / l**2
                                
        #Calculate derivative 
        der = ks.transpose(0,2,1) @ alpha 
        
        #Undo normalization
        der *= y_train_std
        
        #Return derivative
        return der
