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
from categorical_from_binary.hmc.core import (
    CategoricalModelType,
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.hmc.generate import (
    generate_categorical_data_with_covariates_using_multilogit_link,
)
from categorical_from_binary.hmc.report import report_on_experiment_with_covariates


# Configs
random_seed = 42
num_samples = 1024
num_categories = 3
num_covariates_not_counting_bias = 3

# Generate (intercepts-only) categorical data
(
    y_train__one_hot_NK,
    y_test__one_hot_NK,
    x_train_NM,
    x_test_NM,
) = generate_categorical_data_with_covariates_using_multilogit_link(
    num_samples,
    num_categories,
    num_covariates_not_counting_bias,
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
    x_train_NM,
    random_seed,
)

report_on_experiment_with_covariates(
    categorical_model_type,
    betas_SKM_by_N,
    y_test__one_hot_NK,
    x_test_NM,
)
