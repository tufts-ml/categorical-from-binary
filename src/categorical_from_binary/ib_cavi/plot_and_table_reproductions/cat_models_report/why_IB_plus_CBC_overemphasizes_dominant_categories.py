"""
The goal of this plot is to suggest why, in the intercepts-only case, IB+CBC tends to 
warp true category probabilities such that the dominant categories obtain extra probability
mass.  Basically, we compare the two different link functions.  
"""

from functools import partial

import numpy as np
import scipy
from matplotlib import pyplot as plt


Phi_inv = scipy.stats.norm.ppf

# these functions below reference the MLE's for IB, CBC, CBM.  See the MLE equations
# in the UAI-TPM workshop paper

CBC_arg_from_IB_arg_and_constraint = (
    lambda p, constraint: constraint * p / (1 + constraint * p)
)
CBM_arg_from_IB_arg_and_constraint = lambda p, constraint: constraint * p


def make_CBC_arg_from_IB_arg(constraint):
    return partial(CBC_arg_from_IB_arg_and_constraint, constraint=constraint)


def make_CBM_arg_from_IB_arg(constraint):
    return partial(CBM_arg_from_IB_arg_and_constraint, constraint=constraint)


functions_by_models = {"CBC": make_CBC_arg_from_IB_arg, "CBM": make_CBM_arg_from_IB_arg}

constraint_vec = [0.1, 1, 10]

fig, axes = plt.subplots(nrows=len(constraint_vec), ncols=2)


for m, (model, func) in enumerate(functions_by_models.items()):
    for (i, constraint) in enumerate(constraint_vec):
        us = np.linspace(0, 1, 100)
        g = func(constraint)
        xs = [Phi_inv(g(u)) for u in us]  # (S)CBC estimates of beta
        ys = [Phi_inv(u) for u in us]  # IB estimates of beta

        axes[i, m].plot(xs, ys)
        axes[i, m].plot(xs, xs, "--")
        axes[i, m].set_title(f"normalization constraint={constraint}")
fig.supylabel("IB estimates of beta")
fig.suptitle(f"IB estimates of beta by CBC (left) / CBM (right) estimates")
fig.tight_layout(pad=1.0)
plt.show()
