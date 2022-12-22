"""Implements Bayesian linear regression as described in C.M. Bishop. Pattern
recognition and machine learning. Springer, 2006. Helper file for 
`regression.py`.
"""

from iminuit import Minuit
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures

class Polynomial(object):
    def __init__(self, degree=2):
        self.degree = degree
        
    def featurize(self, x):
        return PolynomialFeatures(degree=self.degree).fit_transform(
            np.atleast_2d(x)
        )

class GaussianPolynomial(Polynomial):
    def __init__(self, degree=2, sigma=10.):
        super().__init__(degree=degree)
        self.Mu = np.zeros((self.degree+1,))
        self.Cov = np.diag([100,10,.1,.001][:(self.degree+1)])**2
        self.sigma = sigma
        
    def f(self, x, *Beta):
        return self.featurize(x)@np.array(Beta)
        
    def ado(self):
        m = Minuit(lambda d: -self.pseudoU(d), d=np.random.uniform(high=100))
        m.limits["d"] = (0,100)
        m.simplex().migrad()
        return m.values["d"]
    
    def log_likelihood(self, x, y):
        x = self.featurize(x)
        ll = np.diag(norm.pdf(y, loc=x@self.Mu, 
                              scale=np.diag(self.sigma**2 + x@self.Cov@x.T)**(1/2)
                             ))
        return np.log(ll).sum()
    
    def pseudoU(self, d):
        x = self.featurize(d)[0,:]
        return x@self.Cov@x.T
    
    def update(self, x, y):
        x = self.featurize(x)
        Sn = np.linalg.inv(np.linalg.inv(self.Cov) + \
                (1/self.sigma**2)*x.T@x)
        mn = Sn @ (np.linalg.inv(self.Cov)@self.Mu[:,None] + \
                (1/self.sigma**2)*x.T@np.atleast_2d(y))
        self.Mu = mn[:,0]
        self.Cov = Sn
