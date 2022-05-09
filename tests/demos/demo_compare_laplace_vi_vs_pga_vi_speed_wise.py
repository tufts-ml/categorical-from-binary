
# flake8: noqa

"""
This is a demo,  to be run in real-time via ipython (due to the use of magic commands),
to compare two different methods for approximating the posterior of Bayesian logistic regression:
1) Laplace variational inference
2) Variational inference with polya gamma augmentation
"""

import numpy as np
from pypolyagamma import PyPolyaGamma
from sklearn.linear_model import LogisticRegression
import scipy
from typing import NamedTuple

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_logistic_regression_dataset,
)
from categorical_from_binary.polya_gamma_vi.bayes_logreg.inference import (
    run_polya_gamma_variational_inference_for_bayesian_logistic_regression,
    PriorParameters,
)

from categorical_from_binary.laplace_vi.bayes_logreg.inference import (
    get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi,
)


class Experiment(NamedTuple):
    n_samples : int 
    n_features : int 
    convergence_criterion : float 

# EXPERIMENTS 
experiment_1 = Experiment(n_samples=1000,  n_features = 10, convergence_criterion=0.05)
experiment_2 = Experiment(n_samples=5000,  n_features = 500, convergence_criterion=0.05)

#SETUP 
experiment = experiment_1
dataset = generate_logistic_regression_dataset(experiment.n_samples, experiment.n_features)
beta_dim = len(dataset.beta)
beta_init = np.zeros(beta_dim)
prior_mean = np.zeros(beta_dim)
prior_cov = np.eye(beta_dim)
prior_params = PriorParameters(prior_mean, prior_cov)

# PGA 
%timeit -r 1 -n 1 run_polya_gamma_variational_inference_for_bayesian_logistic_regression( dataset, prior_params, verbose=True, convergence_criterion_drop_in_elbo=experiment.convergence_criterion) # noqa 

# LAPLACE VI 
%timeit -r 1 -n 1 get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi(beta_init, dataset, prior_mean, prior_cov) # noqa 

"""
Conclusions:
    PGA is 
        - faster (a bit, but the difference seems to disappear as the dataset size scales)
        - doesn't require external call to numerical optimizer
            (and therefore a need to consider implementational details,  e.g. choice of optimizer -- BFGS? Nelder Mead? etc.,
            computation of gradient and possibly hessian,  etc.)

Limitations of demo:
    - Need to compute computational complexity for each optimizer
    - Need to make comparison fairer (for Laplace VI) by aligning convergence criterion 
    - Need to make comparison fairer (for PGA VI) by having both codebases equally well-optimized 
        (scipy.optimize, the core of Laplace VI, should be better optimized code than my own base Python)
"""
