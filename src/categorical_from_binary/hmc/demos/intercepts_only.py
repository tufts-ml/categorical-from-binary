"""
Purpose:
    When fitting an independent binary (IB) model to categorical data, we find that the regression weights
    give a (much) better fit to ground truth data when used within the category probabilities 
    from our categorical from binary via marginalization probit model (CBM) than within those of the original diagonal
    orthant probit model (CBC). 

Naming Convention:
    We use MCH's naming convention here, which is to give the dimensionality of each array
    after the last underscore

References: 
    * https://github.com/tufts-ml/CBC-models/blob/main/experiments/CBCProbitDemo_K2.ipynb
    * Wojnowicz, Aeron, Miller, Hughes, "Easy Variational Inference for 
        Categorical Observations via a New View of Diagonal Orthant Probit Models"
"""

import numpy as np

from categorical_from_binary.hmc.core import (
    CategoricalModelType,
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.hmc.generate import generate_intercepts_only_categorical_data
from categorical_from_binary.hmc.report import report_on_intercepts_only_experiment


# Configs
random_seed = 42
num_samples = 1024
true_category_probs_K = np.asarray([0.05, 0.95])

# Generate (intercepts-only) categorical data
(y_train__one_hot_NK, y_test__one_hot_NK,) = generate_intercepts_only_categorical_data(
    true_category_probs_K,
    num_samples,
    random_seed=random_seed,
)

# Run NUTS
num_warmup, num_mcmc_samples = 3000, 10000
Nseen_list = [2, 16, 128, 1024]
categorical_model_type = CategoricalModelType.IB_PROBIT
betas_SKM_by_N = run_nuts_on_categorical_data(
    num_warmup,
    num_mcmc_samples,
    Nseen_list,
    create_categorical_model,
    categorical_model_type,
    y_train__one_hot_NK,
    random_seed=random_seed,
)

# Check the inferred category frequencies when plugging the regression weights
# into the {CBC, CBM} category probability functions
report_on_intercepts_only_experiment(
    true_category_probs_K, categorical_model_type, betas_SKM_by_N
)
