"""Constructors for the `Model()` objects required by `bado`.

Modified version of code in the pyBAD package, available at https://github.com/sabjoslo/pyBAD.
"""

from samplers import *

class Model(object):
    def __init__(self, f, prior, nparams=None, param_bounds=None, p_m=1.,
                 sampler=KDE, **kwargs
                ):
        self._f = f
        self.p_m = p_m
        self.dist = sampler(
            likelihood=self.likelihood_fixed_param, prior=prior, 
            nparams=nparams, param_bounds=param_bounds, **kwargs
        )
        signature = "{}->(n)".format(",".join(["()"] * self.dist.nparams))
        self._vectorized_over_params = np.vectorize(
            self._f, signature=signature, excluded=[self.dist.nparams]
        )
    
    @property
    def p_m(self):
        return self._p_m
    
    @p_m.setter
    def p_m(self, value):
        self._p_m = value
    
    def predict(self, d, ny):
        pred = self._vectorized_over_params(*self.dist.samples.T, d)
        self.pred = np.repeat(pred[:,:,None], ny, axis=-1)
        
class BinaryClassModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _likelihood_unreduced(self, y):
        ll = self.pred.copy()
        ll[:,y==0] = 1-ll[:,y==0]
        return ll
    
    def likelihood(self, y):
        ll = self._likelihood_unreduced(y)
        return np.prod(ll, axis=(1,2))@self.dist.W
        
    def log_likelihood(self, y):
        ll = self._likelihood_unreduced(y)
        return np.log(ll + np.finfo(float).eps).sum(axis=(1,2))@self.dist.W
    
    def likelihood_fixed_param(self, theta, y, d):
        ll = self._f(*theta, d)[:,None]
        ll = np.repeat(ll, y.shape[-1], axis=-1)
        ll[y==0] = 1-ll[y==0]
        return np.prod(ll)
