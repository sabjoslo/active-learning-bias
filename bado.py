"""Implements functions for performing numerical Bayesian inference and adaptive 
designs. The user-facing functions are:
- `U`: Implements a numerical search for the Bayesian optimal design. Requires 
that at least one instance of `models.Model()` be included in the list of 
`args`. Outputs the numerically-estimated utility for each design in 
`design_grid`.
- `update_models`: Updates the parameter distribution of each `Model()` passed
as `args` on the basis of observation `y` given `d`.

Modified version of code in the pyBAD package, available at https://github.com/sabjoslo/pyBAD.
"""

import numpy as np

def _update_p_m(pm, py, ii, nanproof=True):
    if pm[ii] == 0.:
        return pm[ii]
    jj = np.delete(np.arange(pm.shape[0]), ii)
    bf = np.where(
        (py[jj,:,:] != 0) | (py[ii,:,:] != 0), py[jj,:,:] / py[ii,:,:], 1
    )
    # If the likelihood corresponding to a particular model is `nan`, that
    # should indicate that all observed response patterns were impossible
    # under that model.
    if nanproof:
        bf[np.isnan(bf)] = 0.
    # This should prevent `nan`s when multiplying `inf` by 0.
    bf[pm[jj] == 0.,:,:] = 0.
    return pm[ii] / (pm[ii] + (pm[jj,None,None] * bf).sum(axis=0))

update_p_m = np.vectorize(_update_p_m, excluded=[0,1])

def u_parameterEstimation(pm, py, ptheta):
    uu = np.zeros((pm.shape[0],py.shape[-1]))
    for ii, p in enumerate(pm):
        mpy = (py[ii,:,:,:] * ptheta[ii,None,:,None]).sum(axis=1)[:,None,:]
        u_ii = np.log(py[ii,:,:,:] / mpy)
        u_ii = np.where(py[ii,:,:,:] > 0, py[ii,:,:,:] * u_ii, 0).sum(axis=0)
        uu[ii] += u_ii.T@ptheta[ii,:]
    return pm[pm > 0.] @ uu[pm > 0.]

def U(designs, u, *models):
    weights = np.stack([ m.dist.W for m in models ])
    pm = np.array([ m.p_m for m in models ])    
    py = np.zeros((
        len(models),2,models[0].dist.samples.shape[0],designs.shape[0]
    ))
    for mi, m in enumerate(models):
        p1 = m._vectorized_over_params(*m.dist.samples.T, designs)[None,:,:]
        py[mi,:,:,:] += np.vstack((1-p1, p1))
    return u(pm, py, ptheta=weights)

def update_models(y, x, *models):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    for m in models:
        m.predict(x, 1)
    ll = [ m.likelihood(y*np.ones((1,1))) for m in models ]
    pm = np.array([ m.p_m for m in models ])
    py = np.array(ll)[:,None,None]
    # Model probability updates will not affect results reported in the paper,
    # which deal exclusively with parameter estimation on the basis of a single
    # model
    post_p_m = update_p_m(pm, py, np.arange(len(models)))
    for ii, m in enumerate(models):
        m.dist.update(y, x)
        if np.all(np.isnan(m.dist.W)):
            post_p_m[ii] = 0.
    for ii, m in enumerate(models):
        m.p_m = post_p_m[ii]
