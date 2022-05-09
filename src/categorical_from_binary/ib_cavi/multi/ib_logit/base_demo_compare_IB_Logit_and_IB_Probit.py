"""
Here we demo the predictive performance of the IB-Logit over time, 
comparing it to IB-Probit.   In both cases, we use the IB approximation,
and we do approximate Bayesian inference with mean-field VI. 
"""

import pandas as pd

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_logit.inference import (
    compute_multiclass_logit_vi_with_polya_gamma_augmentation,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


pd.set_option("display.max_columns", None)


###
# Construct dataset
###
n_categories = 3
n_features = 3
n_samples = 1000
include_intercept = True
link = Link.MULTI_LOGIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=2.0
)
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link,
    seed=None,
    include_intercept=include_intercept,
)


# Prep training / test split
n_train_samples = int(0.8 * n_samples)
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

####
# Variational Inference: Configs
####

max_n_iterations = 8

# Variational Inference with CBC-LOGIT
print("---Variational inference with CBC-LOGIT---")
results_logit = compute_multiclass_logit_vi_with_polya_gamma_augmentation(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=max_n_iterations,
)

# Variational Inference with CBC-PROBIT
print("---Variational inference with CBC-PROBIT---")
results_probit = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=max_n_iterations,
)


holdout_performance_dict = {
    ("Accuracy", "IB-Logit"): results_logit.holdout_performance_over_time[
        "correct classification rate"
    ],
    ("Accuracy", "IB-Probit"): results_probit.holdout_performance_over_time[
        "correct classification rate"
    ],
    (
        "Mean Log Likelihood",
        "IB-Logit+CBC",
    ): results_logit.holdout_performance_over_time[
        "mean holdout log likelihood for CBC_LOGIT"
    ],
    (
        "Mean Log Likelihood",
        "IB-Probit+CBC",
    ): results_probit.holdout_performance_over_time[
        "mean holdout log likelihood for CBC_PROBIT"
    ],
    (
        "Mean Log Likelihood",
        "IB-Logit+CBM",
    ): results_logit.holdout_performance_over_time[
        "mean holdout log likelihood for CBM_LOGIT"
    ],
    (
        "Mean Log Likelihood",
        "IB-Probit+CBM",
    ): results_probit.holdout_performance_over_time[
        "mean holdout log likelihood for CBM_PROBIT"
    ],
}

holdout_performance = pd.DataFrame(holdout_performance_dict)
pd.options.display.width = 200
print(f"Holdout accuracy by iterate:\n {holdout_performance}")
