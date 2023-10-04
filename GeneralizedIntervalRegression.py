# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:38:00 2022

@author: jacobtri
"""

# where do I download packages? how about here...
import numpy as np
from scipy.stats import norm

class normal:
    self.params = ['mu', 'sigma']
    #self.parent = 'sgt'
    
    def pdf():
        return norm.pdf

class GeneralizedIntervalRegressor:
    
    
    def __init__(self, dist=normal):
        self.dist = dist # be able to access params list to verify opt compatibility, pdf and cdf
        
        
    def index_types(self): # if I can't use this here bc self.y1 not yet defined, then index_types(y1, y2) or just include in fit(...)
        # refer to function to assign indices, ie point, interval etc
        # or just do it here because it can't go before fit (includes Y)
        # todo here: also verify no missings in each line
        # note: break from stata, instead of missing==., use np.inf
        # might need to use np.logical_and() here...?
        # consider import isinf suite directly as to not call np. before each
        self.idx_point = self.y1==self.y2 # and not infinite
        self.idx_interval = self.y1<self.y2 # & not infinite
        self.idx_bottomcoded = np.isneginf(self.y1) and np.isfinite(self.y2)
        self.idx_topcoded = np.isfinite(self.y1) and np.isposinf(self.y2)
        
        
    def fit(self, y, x=None, sigma=None, lambda_=None, p=None, q=None): # add other params, ie p q; cap for Matrix names?
        # Break up two-column Y into distinct col vectors
        self.y1, self.y2 = [*y.T]
        self.N = len(self.y1) # number of observations
        self.usedparams = [f'{param}' if param is not None for param in [x, sigma, lambda_, p, q]]

        
        for pname in ['mu', 'sigma', 'lambda_', 'p', 'q']:
            if pname is not None and f'{pname}' in self.dist.params:
                except NameError:
                    print(f'{pname} not a parameter of selected distribution')
        
        loglike[idx_point] = dist.pdf(self.y1, self.x, self.sigma, self.p, self.q, self.lambda_)
        
        
        
        
    def loglike_point(self, data): #data*params...see statsmodels
        return data.map(self.dist.pdf)
    
    def loglikeobs(self, data):
        # Point
        loglikeobs_point = loglike_point(data[idx_point])
        # Interval and so on...
        loglikeobs = np.sum(loglikeobs_point)