"""Implements a simplified version of Cumulative Prospect Theory [Tversky, A. 
and Kahneman, D. (1992). Advances in Prospect Theory: Cumulative Representation 
of Uncertainty, *Journal of Risk and Uncertainty, 5*].
"""

import numpy as np

# Probability weighting function developed by Prelec, D. (1998). The Probability
# Weighting Function. *Econometrica, 66*(3).
def prelec(p, gamma):
    # Avoid floating point errors
    p = np.minimum(p, 1.)
    return np.exp(-(-np.log(p))**gamma)

def cpt_value(X, p, pwfunc, alpha, lambda_, gamma):
    V = X.copy()
    ipos, jpos = np.where(X >= 0.)
    ineg, jneg = np.where(X < 0.)
    V[ipos,jpos] = X[ipos,jpos]**alpha
    V[ineg,jneg] = -lambda_*(-X[ineg,jneg])**alpha
    w = pwfunc(p, gamma)
    return (V*w).sum(axis=1)

def p1(V_1, V_2, epsilon):
    return 1 / (1 + np.exp(-epsilon * (V_1 - V_2)))
