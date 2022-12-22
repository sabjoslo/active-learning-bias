"""Implements two posterior representations:
- A grid representation (used in preference learning example)
- An importance-sampling mechanism that relies on a weighted density estimate of
samples from the prior (used in classification example)

Modified version of code in the pyBAD package, available at https://github.com/sabjoslo/pyBAD.
"""

from copy import deepcopy
import logging
import numpy as np
from scipy.stats import gaussian_kde

def _in_bounds(x, bounds):
    return x >= bounds[0] and x <= bounds[1]
in_bounds = np.vectorize(_in_bounds, signature="(),(n)->()")
array_in_bounds = np.vectorize(in_bounds, excluded=[1], signature="(n)->(n)")

class FlexiArray(object):
    def __init__(self):
        self.values = np.empty((0,0))
    
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, value):
        self._values = value
    
    def __iadd__(self, x):
        if self._values.shape[0] == 0:
            self._values = x
        else:
            self._values = np.append(self._values, x, axis=0)
        return self
    
    def asarray(self):
        return self._values

class Sampler(object):
    def __init__(
        self, likelihood, prior, init=True, nparams=None, nsamples=10000,
        param_bounds=None, weighted=True
    ):
        self.distribution = deepcopy(prior)
        self.is_prior = True
        self.likelihood = likelihood
        self.nparams = nparams
        self.nsamples = nsamples
        self.param_bounds = param_bounds
        self.prior = prior
        self.weighted = weighted
        self.X = FlexiArray()
        self.Y = FlexiArray()
        if init:
            self.samples, self.W = self.sample()
        
    @property
    def distribution(self):
        return self._distribution
    
    @distribution.setter
    def distribution(self, value):
        self._distribution = value
        
    @property
    def is_prior(self):
        return self._is_prior
    
    @is_prior.setter
    def is_prior(self, value):
        self._is_prior = True
        
    @property
    def likelihood(self):
        return self._likelihood
    
    @likelihood.setter
    def likelihood(self, value):
        self._likelihood = value
        
    @property
    def nparams(self):
        return self._nparams
    
    @nparams.setter
    def nparams(self, value):
        self._nparams = value
        if isinstance(self._nparams, type(None)):
            _rvs = self._distribution.rvs()
            if hasattr(_rvs, "__iter__"):
                self._nparams = _rvs.shape[0]
            else:
                logging.info("""Setting `Sampler.nparams` = 1.
                Manually pass `nparams` as argument to override.""")
                self._nparams = 1
    
    @property
    def nsamples(self):
        return self._nsamples
    
    @nsamples.setter
    def nsamples(self, value):
        self._nsamples = value
    
    @property
    def param_bounds(self):
        return self._param_bounds
    
    @param_bounds.setter
    def param_bounds(self, value):
        if isinstance(value, type(None)):
            self._param_bounds = np.vstack((np.repeat(-np.inf, self._nparams),
                                            np.repeat(np.inf, self._nparams)
                                           )).T
        else:
            self._param_bounds = np.array(value)
    
    @property
    def prior(self):
        return self._prior
    
    @prior.setter
    def prior(self, value):
        self._prior = value
    
    @property
    def samples(self):
        return self._samples
    
    @samples.setter
    def samples(self, value):
        self._samples = value
    
    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, value):
        self._W = value
        
    @property
    def weighted(self):
        return self._weighted
    
    @weighted.setter
    def weighted(self, value):
        self._weighted = value
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, value):
        self._X = value
    
    @property
    def Y(self):
        return self._Y
    
    @Y.setter
    def Y(self, value):
        self._Y = value
    
    def parameter_likelihood(self, y, d, samples=None):
        if isinstance(samples, type(None)):
            samples = self._samples
        return np.apply_along_axis(self._likelihood, 1, samples, y, d)
        
    # Pseudo- (potentially unnormalized) pdf for the distribution
    def _ppdf(self, x):
        return self._distribution.pdf(x)
        
    def _prior_weights(self, y, d, zeroproof=True):
        ll = self.parameter_likelihood(y, d)
        wghts = ll * self._W
        if wghts.sum() == 0. and zeroproof:
            wghts += 1.
        return wghts / wghts.sum()
    
    def _sample(self):
        return self._distribution.rvs(size=(self._nsamples,self._nparams))
    
    def sample(self, throwaway=False):
        samples = self._sample()
        wghts = self._set_weights(
            samples, weighted=(~self._is_prior & self._weighted)
        )
        if not throwaway:
            self.samples = samples
            self.W = wghts
        return samples, wghts
        
    def sample_from_prior(self):
        return self._prior.rvs(size=(self._nsamples,self._nparams))
    
    def _set_equal_weights(self, samples):
        return np.ones_like(samples[:,0]) / samples.shape[0]
    
    # OK to ignore normalizing constants for P and Q
    # [ (P*a) / (Q*b) ] / \sum_{ (P*a) / (Q*b) } = ( P / Q ) / \sum_{ P / Q }
    def _set_importance_weights(self, samples, Q=None):
        if isinstance(Q, type(None)):
            Q = self._ppdf(samples)
        P = self._prior.pdf(samples).prod(axis=1)
        P *= self.parameter_likelihood(
            self._Y.asarray(), self._X.asarray(), samples=samples
        )
        wghts = P / Q
        wghts[Q == 0.] = 0.
        return wghts / wghts.sum()
    
    def _set_weights(self, samples, Q=None, weighted=True):
        if not weighted:
            return self._set_equal_weights(samples)
        return self._set_importance_weights(samples, Q=Q)
    
    def update(self, y, d):
        self._X += d
        self._Y += y
        self._distribution, self._samples, self._W = self._update(y, d)
        self._is_prior = False
    
class Grid(Sampler):
    def __init__(self, **kwargs):
        _init = kwargs.get("init", True)
        kwargs["init"] = False
        self._res = kwargs.get("res", None)
        if "res" in kwargs.keys():
            del kwargs["res"]
        super().__init__(**kwargs)
        if isinstance(self._nsamples, type(None)):
            assert isinstance(self._res, int)
            self.nsamples = self._res**self._nparams
        if isinstance(self._res, type(None)):
            assert isinstance(self._nsamples, int)
            _res = self._nsamples**(1./self._nparams)
            assert _res == int(_res)
            self._res = int(_res)
        assert self._nsamples == self._res**self._nparams
        self._quantiles = np.linspace(
            np.finfo(float).eps, 1-np.finfo(float).eps, self._res
        )
        qstack = self._stack_list([self._quantiles]*self._nparams)
        self._grid = self._distribution.ppf(qstack)
        if _init:
            self.samples, self.W = self.sample()
        
    def _ppdf(self, x):
        return self._distribution.pdf(x).prod(axis=1)
        
    def _sample(self):
        return self._grid
    
    def _set_weights(self, samples, weighted=None):
        return self._set_equal_weights(samples)
    
    def _stack_list(self, alist):
        grid = np.meshgrid(*np.stack(alist))
        return np.stack([ g.ravel() for g in grid ], axis=1)
    
    def _update(self, y, d):
        ll = self.parameter_likelihood(y, d)
        if ll.sum() == 0.:
            return self._distribution, self._samples, self._W
        wghts = self._W * ll
        return self._distribution, self._samples, wghts / wghts.sum()
    
class KDE(Sampler):
    def __init__(self, **kwargs):
        self.bw_method = kwargs.get("bw_method")
        if "bw_method" in kwargs.keys():
            del kwargs["bw_method"]
        super().__init__(**kwargs)
    
    def _ppdf(self, x):
        return self._distribution.pdf(x.T)
    
    def _resample_kde(self, kde=None):
        if isinstance(kde, type(None)):
            kde = self._distribution
        theta = kde.resample(size=self._nsamples).T
        out_of_bounds = ~array_in_bounds(theta, self._param_bounds)
        while np.any(out_of_bounds):
            logging.info("""{}% of samples are out of bounds. 
            Resampling...""".format(
                np.sum(out_of_bounds) / np.prod(out_of_bounds.shape) * 100
            ))
            ridx, cidx = np.where(out_of_bounds)
            theta[ridx,:] = kde.resample(size=ridx.shape[0]).T
            out_of_bounds = ~array_in_bounds(theta, self._param_bounds)
        return theta
    
    def _sample(self, n=100):
        if self._is_prior:
            return self.sample_from_prior()
        theta = self._distribution.resample(size=n).T
        out_of_bounds = ~array_in_bounds(theta, self._param_bounds)
        while np.any(out_of_bounds):
            logging.info("""{}% of samples are out of bounds. 
            Resampling...""".format(
                np.sum(out_of_bounds) / np.prod(out_of_bounds.shape) * 100
            ))
            ridx, cidx = np.where(out_of_bounds)
            theta[ridx,:] = self._distribution.resample(size=ridx.shape[0]).T
            out_of_bounds = ~array_in_bounds(theta, self._param_bounds)
        return theta
        
    def _update(self, y, d):
        # Add epsilon to prevent collapse when encountering "impossible"
        # observations
        w = self._prior_weights(y, d)
        post = gaussian_kde(
            self._samples.T, bw_method=self.bw_method, weights=w
        )
        samples = self._resample_kde(kde=post)
        return post, samples, self._set_weights(
            samples, Q=post.pdf(samples.T), weighted=self._weighted
        )
