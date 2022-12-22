#!/usr/bin/env python

"""Script to run simulations reported in section 6.4. Each run performs one 
"experiment." Takes command line arguments `idx` and `designf` (in order).
- `idx` determines the parameter settings of the true model. The integer passed
indexes the parameter file `par_cl`.
- `designf`. One of { "adaptive", "random" }. Determines whether the experiment
uses adaptive or random sampling.

Experiments produced using this script can be analyzed in the notebook
`classification.ipynb` to reproduce Fig 7.
"""

from argparse import ArgumentParser
import datetime
import numpy as np
import os
import pickle
from scipy.stats import bernoulli, norm
from bado import *
from models import BinaryClassModel

# Determines where simulation data files are written
ddir = "classification"
# Number of trials experiments are run for
ntrials = 100
# epsilon corresponding to the noise of the hypothesized model class
## Rerun with `epsilon = .1` for results in Fig 7b
## Rerun with `epsilon = .01` for results in Fig 7c
epsilon = 1.
#epsilon = .1
#epsilon = .01

design_grid = np.linspace(0, 100, 101)[:,None]

logistic = lambda x: 1 / (1 + np.exp(-x)) 

altModel = lambda b0, b1, x: logistic(epsilon*(b0 + b1*x[:,0]))

def trueModel(b0, b1, b2, x):
    return logistic(b0 + b1*x[:,0] + b2*x[:,0]**2)

def adaptive_design(m):
    return design_grid[U(design_grid, u_parameterEstimation, m).argmax(),0]

def random_design(m):
    return design_grid[np.random.randint(100),:]
        
def simulation(paridx, designf=adaptive_design, nsamples=10000):
    true_params = parc[paridx,:]
    m = BinaryClassModel(
        f=altModel, prior=norm(loc=np.zeros((2,)), scale=[100,10]), p_m=1.
    )
    D = []
    LL = []
    for trial in range(ntrials):
        with open(f"{args.designf}-{paridx}.log", "a") as log:
            log.write(f"[{datetime.datetime.now()}]: At trial {trial}\n")
        d = designf(m)
        D.append(d)
        y = bernoulli.rvs(trueModel(*true_params, np.atleast_2d(d)))
        update_models(y, d, m)
        _d = np.random.uniform(high=100, size=100)[:,None]
        _y = bernoulli.rvs(trueModel(*true_params, _d))[:,None]
        m.predict(_d, 1)
        LL.append(m.log_likelihood(_y))
    return D, LL, m.dist.samples, m.dist.W

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("designf", type=str)
    args = parser.parse_args()
    
    designf = dict(adaptive=adaptive_design, random=random_design)[args.designf]
    
    parc = pickle.load(open("par_cl", "rb"))
    D, LL, samples, weights = simulation(args.idx, designf=designf)
    
    with open(f"{ddir}/{args.designf}-{args.idx}.pkl", "wb") as wfh:
        pickle.dump(dict(D=D, LL=LL, samples=samples, weights=weights), wfh)
        
    os.remove(f"{args.designf}-{args.idx}.log")
