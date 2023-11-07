# -*- coding: utf-8 -*-
"""
TARGET
------
import GeneralizedIntervalRegressor as gintreg

data = pd.DataFrame(...)

model = gintreg(
    data['y1'], 
    data['y2'], 
    mu=data[['x1','x2','x3']],
    distribution='skewed normal'
    )
results = model.fit(method='dfp')
results.plot()
print(results)


NOTES
-----
* expects 'missing values' to be np.nan
* I (jacob) changed Distribution.pdf-->logpdf, cdf-->logcdf. This was because 
  norm.pdf was way too sensitive to 'bad' guesses; norm.pdf would often return
  0.0, and log(0.0) is undefined, so it would break. I want to make a note here
  because I don't know how many of our distributions' pdf/cdf are logged already.

TODO (9/11/2023)
----
[DONE 9/20/2023] test code so far and debug as necessary
[DONE 9/20/2023] related to above -- indexing arrays ([row,col]?, [col,row]?) and np.dot() need looked at
built fit(): ...
    - estimate starting parameters
    - pass through arguments (e.g. method)
    - anything else?
rework results summary display to show estimate names (e.g. mu_cons, mu_x1, ..., sigma_cons, sigma_x1,...)
investigate custom progress readout for scipy.min
postestimation commands (easy)
    - gini
    - aic / bic 
    - plot (calculate parameters as estimates*covariates@means)
"""

################### IMPORT PACKAGES #########################
import numpy as np
import math as math
from scipy.stats import norm, gamma
from scipy.optimize import minimize 
from statsmodels.tools.tools import add_constant

########### IMPORT PACKAGES FOR TESTING #############
from scipy.stats import skewnorm

########### Functions for Evaluation #############
sign = lambda x: math.copysign(1, x)

################### DISTRIBUTION LIBRARY ####################
class Distribution:
    
    parameters = None
    
    def logcdf(): 
        raise NotImplementedError
    
    def logpdf():
        raise NotImplementedError 


class Normal(Distribution):
    
    parameters = ['mu','sigma']
    
    def logcdf(y,mu,sigma):
        return norm.logcdf(y,mu,sigma)
    
    def logpdf(y,mu,sigma):
        return norm.logpdf(y,mu,sigma)


class NormalUnitVariance(Distribution): # example of a 'special case' by constraining parameter(s)
    
    parameters = ['mu']
     
    def logcdf(y,mu):
        return Normal.logcdf(y,mu,sigma=1)
    
    def logpdf(y,mu):
        return Normal.logpdf(y,mu,sigma=1)
     
class SkewedNormal(Distribution): 
    
     parameters = ['mu', 'sigma', 'lam', 'p']
     
     def logcdf(y,mu,sigma,lam, p=2): #working on building these fns out
        lam = (np.exp(lam) - 1)/ (np.exp(lam)+1)
        z = (abs(y - mu)**p)/((np.exp(sigma)**p)*(1+lam*sign(y - mu))**p)
        return .5*(1-lam) + .5*(1 + lam*sign(y-mu))*sign(y-mu)*gamma.pdf(x=z, a=(1/p))
     
     def logpdf(y,mu,sigma,lam, p=2):
        lam = (np.exp(lam) - 1)/ (np.exp(lam)+1)
        x = y - mu
        s = np.log(p) - (abs(x)** p / (np.exp(sigma)*(1+lam*sign(x)))**p)
        l = np.log(2) + sigma + math.lgamma((1/p))
        return s - l



################### ESTIMATOR ###############################
class GeneralizedIntervalRegressor():

    def __init__(self, 
                   y1,
                   y2, 
                   mu=None,
                delta=None,
                sigma=None,
                  lam=None,
                    p=None,
                    q=None,
         distribution='normal',
                 ):
        
        self.y1 = y1
        self.y2 = y2

        distributions = {
            'normal':Normal,
            'snormal': SkewedNormal
            }
        self.dist = distributions[distribution]
        
        # Add a constant to each (used) covariate and store in a dictionary
        _covariates = {'mu':mu, 'delta':delta, 'sigma':sigma, 'lam':lam, 'p':p, 'q':q}
        covariates = {}
        for parameter in self.dist.parameters:
            if _covariates[parameter] is None:
                covariates[parameter] = np.ones((len(y1),1))
            else: 
                covariates[parameter] = add_constant(_covariates[parameter],prepend=True)
        self.covariates = covariates
        
        # Checks
        if invalid:=sum(y1>y2):
            raise ValueError(f'y1 greater than y2 in {invalid} observations')
            
        if invalid:=sum(np.logical_and(np.isnan(y1), np.isnan(y2))):
            raise ValueError(f'y1 and y2 are both missing in {invalid} observations')
           
        for key,value in _covariates.items():
            if key not in self.dist.parameters and value is not None:
                raise ValueError(f'{key} is not a parameter of the {distribution} distribution')
                
        if distribution in ['lognormal','weibull','gamma','ggamma','burr-3','burr-12','gb2']:
            if invalid:=sum(y1<0):
                raise ValueError(f'{distribution} is nonnegative-valued but y is negative in {invalid} observations')


        # Store indices of data type (point, left- or right-censored, interval)
        self.point = y1==y2
        self.lcens = np.isnan(y1)
        self.rcens = np.isnan(y2)
        self.intvl = np.logical_not(np.logical_or.reduce((self.point, self.lcens, self.rcens)))
        
        # Prep work for unpack()
        cutpoints = [0]
        sizes = [c.shape[1] for c in covariates.values()]
        for size in sizes:
            cutpoints.append(cutpoints[-1] + size)
        self.partitions = [(a,b) for a,b in zip(cutpoints[:-1], cutpoints[1:])]
        
    
    def unpack(self, estimates):
        return [estimates[a:b] for a,b in self.partitions]
    
    
    def linear_combination_by_parameter(self, estimates):
        cov_est = zip(self.covariates.values(), self.unpack(estimates))
        dotted = [np.dot(cov,est) for cov,est in cov_est]
        return np.array(dotted) # return as array so it can be indexed all at once
    
    
    def _llf_point(self, x):
        y1 = self.y1[self.point]
        x = x[:,self.point]
        return self.dist.logpdf(y1, *x) 
    
    def _llf_lcens(self, x):
        y2 = self.y2[self.lcens]
        x = x[:,self.lcens]
        return self.dist.logcdf(y2, *x)
     
    def _llf_rcens(self, x):
        y1 = self.y1[self.rcens]
        x = x[:,self.rcens]
        return 1 - self.dist.logcdf(y1, *x)
    
    def _llf_intvl(self, x):
        y1 = self.y1[self.intvl]
        y2 = self.y2[self.intvl]
        x = x[:,self.intvl]
        return self.dist.logcdf(y2, *x) - self.dist.logcdf(y1, *x)
    
    def llf(self, estimates): 

        x = self.linear_combination_by_parameter(estimates)
        
        loglike = np.empty_like(self.y1)
        loglike[self.point] = self._llf_point(x)
        loglike[self.lcens] = self._llf_lcens(x)
        loglike[self.rcens] = self._llf_rcens(x)
        loglike[self.intvl] = self._llf_intvl(x)
        
        return -np.sum(loglike)
    
    
    def fit(self):
        ''' use scipy.optimize.minimize(fun=llf, x0=initial, ...) '''
        
        x0 = [1]*self.partitions[-1][-1] # starting values for estimates
        method = 'Nelder-Mead'
        # opts={"maxiter":5000, "maxfev":5000}
        # return minimize(self.llf, x0, method=method, options=opts)
        return minimize(self.llf, x0, method=method, options={"maxiter":500*len(self.point), "maxfev":500*(len(self.point))})
    
    
    
    
    
################ NORM DIST TEST ################## sorry for spaghetti comments!!
# x = np.arange(100)
# e = np.random.normal(size=(100,))
# B0 = 2 # contant is 2
# B1 = 5 # coeff on x is 5
# y = B0 + x*B1 + e # this is the 'truth'; gintreg tries to estimate B0 and B1
# # using only y and x (and the assumption that e is normally distributed)

# model = GeneralizedIntervalRegressor(y, y, mu=x) # simple example: point data, 
# # one covariate affecting location parameter and using the normal distribution.
# starting_estimates = [1,1,1] # guesses for B0,B1,sigma. 
# #print(model.llf(starting_estimates)) # prints loglikihood value for model at starting_estimates
# print(model.fit())

"""
### outcomes ###
sigma is not estimating efficiently, max iterations and max function evaluations adjusted
to allow more consistent convergence, but may want to optimize later for computation efficiency
"""
"""
### to do ###
test rc, lc, interval
"""


################ SNORM DIST TEST ################## sorry for spaghetti comments!!
x = np.arange(100)
# a = skewness parameter, When a = 0 the distribution is identical to a normal distribution
a = 4
e = skewnorm.rvs(a, size=100)
B0 = 2 # contant is 2
B1 = 5 # coeff on x is 5
y = B0 + x*B1 + e # this is the 'truth'; gintreg tries to estimate B0 and B1
# using only y and x (and the assumption that e is skewed normal distributed)

#     parameters = ['mu', 'sigma', 'lam', 'p']
model = GeneralizedIntervalRegressor(y, y, mu=x, distribution='snormal') # simple example: point data, 
# one covariate affecting location parameter and using the normal distribution.
starting_estimates = [1,1,1,1] # guesses for B0,B1,sigma,lamda 
#print(model.llf(starting_estimates)) # prints loglikihood value for model at starting_estimates
print(model.fit())

"""
### outcomes ###
interval data is not estimating efficiently, max iterations and max function evaluations adjusted
to allow more consistent convergence, but may want to optimize later for computation efficiency
"""
a = 4
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
r = skewnorm.rvs(a, size=1000)