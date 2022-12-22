This folder contains all the code necessary to reproduce the figures in Sloman, S. J., Oppenheimer, D. M., Broomell, S. B., and Shalizi, C. R. (2022). *Characterizing the robustness of Bayesian adaptive experimental designs to active learning bias*. [https://arxiv.org/abs/2205.13698](https://arxiv.org/abs/2205.13698). All simulations were run using only CPU power on one of the author's laptops.

# Dependencies

`alb.yml` contains specifications for a *conda* environment that includes all necessary dependencies.

# Bayesian linear regression

**Figures 2 - 5** can be reproduced by first running `regression.py`, and then the notebook `regression.ipynb`. *regression.py* relies on
- `bayes_linreg.py` (implements Bayesian linear regression).

The parameter settings for the generating models can be found in the *pickle* files `par_reg_degree2` (for the quadratic generating model) and `par_reg_degree3` (for the cubic generating model).

The design distributions under Bayesian adaptive design can be found in the *txt* files `design_dist_1` (under the linear hypothesized model; shown in Figure 2a and used for simulations in Figure 5c) and `design_dist_2` (under the quadratic hypothesized model; shown in Figure 2b and used for simulations in Figure 5d).

# Preference learning

**Figure 6** can be reproduced in the notebook `preference_learning.ipynb`, which relies on
- `bado.py` (implements numerical Bayesian adaptive design optimization),
- `models.py` (implements likelihood functions for a binary class model),
- `samplers.py` (implements numerical representations of posterior distributions), and
- `cpt.py` (implements Cumulative Prospect Theory).

# Classification

**Figure 7** can be reproduced by first running `classification.py`, and then the notebook `classification.ipynb`. *classification.py* also relies on `bado.py`, `models.py` and `samplers.py`.

The parameter settings for the generating models can be found in the *pickle* file `par_cl`.
