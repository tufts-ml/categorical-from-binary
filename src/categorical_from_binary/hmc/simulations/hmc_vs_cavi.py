from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.hmc.core import (
    CategoricalModelType,
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.metrics import compute_metrics
from categorical_from_binary.performance_over_time.for_mcmc import (
    construct_performance_over_time_for_MCMC,
)
from categorical_from_binary.timing import time_me


print(
    f"Here is a simulation which is unfavorable to CAVI (I think - we're waiting for it to finish running)"
    f""
)

"""
n_categories = 100
n_sparse_categories = 0
n_features = 200
n_sparse_features = 0
n_samples = 100000
n_train_samples = 80000
include_intercept = True
link_for_data_generation = Link.MULTI_LOGIT
beta_category_strategy=ControlCategoryPredictability(scale_for_predictive_categories=2.0)
scale_for_intercept = 1.0
seed = 0
"""

###
# Construct dataset
###

n_categories = 300
n_sparse_categories = 0
n_features = 600
n_sparse_features = 0
n_samples = 70000
n_train_samples = int(0.8 * n_samples)
include_intercept = True
link_for_data_generation = Link.MULTI_LOGIT
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=2.0
)
scale_for_intercept = 1.0
seed = 0

dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    n_categories_where_all_beta_coefficients_are_sparse=n_sparse_categories,
    n_sparse_features=n_sparse_features,
    beta_0=None,
    link=link_for_data_generation,
    seed=seed,
    include_intercept=include_intercept,
    beta_category_strategy=beta_category_strategy,
    scale_for_intercept=scale_for_intercept,
)


# Prep training / test split
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

####
# Variational Inference
####
results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=500  # 50,
    # convergence_criterion_drop_in_mean_elbo=0.01,
)
performance_over_time_CAVI = results.performance_over_time

print(f"\n\nCAVI performance over time: \n {performance_over_time_CAVI }")

probs_true = construct_category_probs(
    covariates_test,
    dataset.beta,
    link_for_data_generation,
)
performance_metrics_with_data_generating_process = compute_metrics(
    probs_true, labels_test
)
print(
    f"Performance metrics with data generating process: {performance_metrics_with_data_generating_process}"
)

probs_IB_CAVI_plus_CBC = construct_category_probs(
    covariates_test,
    results.variational_params.beta.mean,
    Link.CBC_PROBIT,
)
metrics = compute_metrics(probs_IB_CAVI_plus_CBC, labels_test)
print(f"{metrics}")

####
# Hamiltonian Monte Carlo
####
num_total = 100

# common stuff
categorical_model_type = CategoricalModelType.CBC_PROBIT
Nseen_list = [n_train_samples]

### Short Warmup
num_warmup = int(0.1 * num_total)
num_post_warmup = num_total - num_warmup
stride_for_evaluating_holdout_performance = 1
beta_samples_HMC_dict_short_warmup, time_for_HMC_short_warmup = time_me(
    run_nuts_on_categorical_data
)(
    num_warmup,
    num_post_warmup,
    Nseen_list,
    create_categorical_model,
    categorical_model_type,
    labels_train,
    covariates_train,
    random_seed=0,
)


beta_samples_HMC_short_warmup = beta_samples_HMC_dict_short_warmup[n_train_samples]
holdout_performance_over_time_HMC_short_warmup = (
    construct_performance_over_time_for_MCMC(
        beta_samples_HMC_short_warmup,
        time_for_HMC_short_warmup,
        covariates_train,
        labels_train,
        covariates_test,
        labels_test,
        Link.CBC_PROBIT,
        stride=stride_for_evaluating_holdout_performance,
        n_warmup_samples=num_warmup,
        one_beta_sample_has_transposed_orientation=True,
    )
)


print(
    f"\n\nHMC holdout performance over time (short warmup): \n {holdout_performance_over_time_HMC_short_warmup }"
)


# ### Long Warmup
# num_warmup = int(0.9 * num_total)
# num_post_warmup = num_total - num_warmup
# stride_for_evaluating_holdout_performance = int(num_post_warmup / 5)
# beta_samples_HMC_dict_long_warmup, time_for_HMC_long_warmup = time_me(run_nuts_on_categorical_data)(
#     num_warmup,
#     num_post_warmup,
#     Nseen_list,
#     create_categorical_model,
#     categorical_model_type,
#     labels_train,
#     covariates_train,
#     random_seed=0,
# )


# beta_samples_HMC_dict_long_warmup = beta_samples_HMC_dict_long_warmup[n_train_samples]
# holdout_performance_over_time_HMC_long_warmup = construct_holdout_performance_over_time_for_HMC(
#     beta_samples_HMC_dict_long_warmup,
#     time_for_HMC_long_warmup,
#     num_warmup,
#     covariates_test,
#     labels_test,
#     Link.CBC_PROBIT,
#     stride=stride_for_evaluating_holdout_performance,
# )


# print(f"\n\nHMC holdout performance over time (long warmup): \n {holdout_performance_over_time_HMC_long_warmup }")
