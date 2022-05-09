import numpy as np


np.set_printoptions(precision=3, suppress=True)

from categorical_from_binary.autodiff.jax_helpers import optimize_beta_for_CBM_model


###
# Generate data
###
# data for K=2, M=3, N=4
# if the first feature is high, then we use category 1. otherwise we use category 2.
X = np.array([[0.1, 0.2, 0.3], [1, 2, 3], [1, 2.1, 3], [10, 2, 3], [20, 2, 3]])
y = np.array(
    [
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ]
)


###
# Find beta MLE using autodiff
###
beta_init = np.zeros((3, 2))
beta_star = optimize_beta_for_CBM_model(beta_init, X, y)
