"""
Here we demo the predictive performance over time.
"""
import pandas as pd

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_ib_cavi_with_normal_prior,
)


###
# Construct dataset
###
n_categories = 3
n_features = 3
n_samples = 5000
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
    beta_category_strategy=beta_category_strategy,
)


# Prep training / test split
n_train_samples = 4000
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

####
# Variational Inference
####
ib_model = IB_Model.PROBIT
results = compute_ib_cavi_with_normal_prior(
    ib_model,
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    convergence_criterion_drop_in_mean_elbo=0.01,
)
pd.set_option("display.max_columns", None)
print(f"\n\nPerformance over time: \n {results.performance_over_time}")
