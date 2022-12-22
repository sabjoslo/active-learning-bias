#!/usr/bin/env python

"""Script to run simulations reported in sections 6.1-6.3. Each run performs one 
"experiment." Takes command line arguments `idx`, `designf`, and `degree` 
(in order).
- `idx` determines the parameter settings of the true model. The integer passed
indexes the parameter file (`par_reg_degree2` or `par_reg_degree3`).
- `designf`. One of { "adaptive", "random" }. Determines whether the experiment
uses adaptive or random sampling.
- `degree`. One of { 1, 2, 3 }. Determines the degree of the hypothesized model
class.

Using just the above command line arguments, the user can reproduce the 
experiments shown in Figs 2 & 3a. The following variables must be changed in the
script itself to reproduce the other regression experiments:
- `sigma_hyp`. Sigma corresponding to the additive error term of the 
hypothesized model class.
- `fstar`. One of { 2, 3 }. Degree of the generating model class (f*).

Experiments produced using this script can be analyzed in the notebook
`regression.ipynb` to reproduce Figs 2-4.
"""

from argparse import ArgumentParser
import datetime
import os
import pickle
import sys
from bayes_linreg import *

# Determines where simulation data files are written
ddir = "regression_sigma100"
# Number of trials experiments are run for
ntrials = 100
# sigma corresponding to the additive error term of the hypothesized model class
## Rerun with `sigma_hyp = 1000.` for results in Fig 5
sigma_hyp = 100.
#sigma_hyp = 1000.
# sigma corresponding to the additive error term of the true model
sigma_true = 100.
# Degree of true model (f*)
## Rerun with `fstar = 3` for results in Figs 4b, 5b & 5d
fstar = 2
#fstar = 3
if fstar == 2:
    trueModel = lambda b0, b1, b2, x: b0 + b1*x + b2*x**2
elif fstar == 3:
    trueModel = lambda b0, b1, b2, b3, x: b0 + b1*x + b2*x**2 + b3*x**3
thetaf = f"par_reg_degree{fstar}"

def adaptive_design(m):
    return m.ado()

# Used for Figures 5c-d
def fixed_design(m, design_dist):
    return np.random.choice(design_dist)

def random_design(m):
    return np.random.uniform(low=0, high=100)

def simulation(true_params, ii, degree=2, designf=adaptive_design):
    logf = f"{args.designf}-{degree}-{ii}.log"
    
    m = GaussianPolynomial(degree=degree, sigma=sigma_hyp)
    D = []
    LL = []
    for trial in range(ntrials):
        with open(logf, "a") as wfh:
            wfh.write(f"[{datetime.datetime.now()}] At trial {trial}\n")
        d = designf(m)
        D.append(d)
        y = norm.rvs(trueModel(*true_params, d), scale=sigma_true)
        m.update(d, y)
        _d = np.random.uniform(high=100, size=100)[:,None]
        _y = norm.rvs(trueModel(*true_params, _d), scale=sigma_true)
        LL.append(m.log_likelihood(_d, _y))
        
    os.remove(logf)
    return D, LL, m.Mu, m.Cov

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("designf", type=str)
    parser.add_argument("degree", type=int)
    args = parser.parse_args()
    
    designf = dict(
        adaptive=adaptive_design, fixed=fixed_design, random=random_design
    )[args.designf]
    
    # `design_dist_1` lists designs selected in simulations where `fstar = 2`,
    # `sigma_hyp = 100.` and `degree = 1`
    # `design_dist_2` lists designs selected in simulations where `fstar = 3`, 
    # `sigma_hyp = 100.` and `degree = 2`
    if args.designf == "fixed":
        design_dist = np.loadtxt(f"design_dist_{args.degree}")
        designf = lambda m: fixed_design(m, design_dist=design_dist)
    
    parc = pickle.load(open(thetaf, "rb"))
    D, LL, Mu, Cov = simulation(
        parc[args.idx,:], args.idx, degree=args.degree, designf=designf
    )
    with open(f"{ddir}/{args.designf}-{args.degree}-{args.idx}.pkl", "wb") as wfh:
        pickle.dump(dict(D=D, LL=LL, Mu=Mu, Cov=Cov), wfh)
