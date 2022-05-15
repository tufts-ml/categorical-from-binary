import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.experiments.posterior_approximation.helpers import (
    BetaSamplesAndLink,
    add_bma_to_cat_prob_data_by_method,
    construct_cat_prob_data_by_method,
    sample_beta_advi,
    sample_beta_cavi,
)
from categorical_from_binary.hmc.core import (
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.multi.inference import IB_Model
from categorical_from_binary.kucukelbir.inference import (
    Metadata,
    do_advi_inference_via_kucukelbir_algo,
)
from categorical_from_binary.timing import time_me


###
# Preliminaries
###

beta_samples_and_link_by_method = dict()

###
# Construct dataset
###
n_categories = 3
n_sparse_categories = 0
n_features = 1
n_samples = 1000
n_train_samples = 900
include_intercept = True
link = Link.MULTI_LOGIT
seed = 0
scale_for_predictive_categories = 2.0  # 2.0
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=scale_for_predictive_categories
)
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    n_categories_where_all_beta_coefficients_are_sparse=n_sparse_categories,
    beta_0=None,
    link=link,
    seed=seed,
    include_intercept=include_intercept,
    beta_category_strategy=beta_category_strategy,
)


# Prep training / test split
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

####
# IB-CAVI Inference
####
results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    convergence_criterion_drop_in_mean_elbo=0.01,
)


####
# NUTS Inference
####

num_warmup, num_mcmc_samples = 300, 1000

Nseen_list = [n_train_samples]
links_for_nuts = [
    Link.MULTI_LOGIT,
    Link.CBC_PROBIT,
]

for link_for_nuts in links_for_nuts:

    beta_samples_NUTS_dict, time_for_nuts = time_me(run_nuts_on_categorical_data)(
        num_warmup,
        num_mcmc_samples,
        Nseen_list,
        create_categorical_model,
        link_for_nuts,
        labels_train,
        covariates_train,
        random_seed=0,
    )
    beta_samples_for_nuts = np.array(beta_samples_NUTS_dict[n_train_samples])  # L x M
    beta_samples_for_nuts = np.swapaxes(beta_samples_for_nuts, 1, 2)  # M x L
    link_for_nuts = Link[link_for_nuts.name]
    name_for_nuts = f"nuts_{link_for_nuts.name}"
    beta_samples_and_link_by_method[name_for_nuts] = BetaSamplesAndLink(
        beta_samples_for_nuts, link_for_nuts
    )

###
# ADVI Inference
###
link = Link.SOFTMAX  # Link.CBC_PROBIT
link_advi = Link.SOFTMAX
lr = 1.0
n_advi_iterations = 400
metadata = Metadata(num_mcmc_samples, n_features, n_categories, include_intercept)

print(f"\r--- Now doing ADVI with lr {lr:.02f} ---")
(
    beta_mean_ADVI_CBC,
    beta_stds_ADVI_CBC,
    performance_ADVI_CBC,
) = do_advi_inference_via_kucukelbir_algo(
    labels_train,
    covariates_train,
    metadata,
    link,
    n_advi_iterations,
    lr,
    seed,
    labels_test=labels_test,
    covariates_test=covariates_test,
)


###
# Construct posterior samples
###
# Need to sample from posterior betas for VI methods
# we already have posterior samples for NUTS

# CAVI
beta_mean_cavi = results.variational_params.beta.mean  # M x K
beta_cov_across_M_for_all_K = (
    results.variational_params.beta.cov
)  # M x M  (uniform across K)
M, L = np.shape(results.variational_params.beta.mean)
beta_samples_cavi = np.zeros((num_mcmc_samples, M, L))
for i in range(num_mcmc_samples):
    beta_samples_cavi[i, :] = sample_beta_cavi(
        beta_mean_cavi, beta_cov_across_M_for_all_K, seed=i
    )
for link_cavi in [Link.CBC_PROBIT, Link.CBM_PROBIT]:
    beta_samples_and_link_by_method[f"ib_cavi_{link_cavi.name}"] = BetaSamplesAndLink(
        beta_samples_cavi, link_cavi
    )


# ADVI
M, L = np.shape(beta_mean_ADVI_CBC)
beta_samples_advi = np.zeros((num_mcmc_samples, M, L))
for i in range(num_mcmc_samples):
    beta_samples_advi[i, :] = sample_beta_advi(
        beta_mean_ADVI_CBC, beta_stds_ADVI_CBC, seed=i
    )
beta_samples_and_link_by_method[f"advi_{link_advi.name}"] = BetaSamplesAndLink(
    beta_samples_advi, link_advi
)


###
# Get category probability samples for a FIXED covariate vector
###

sample_idx = 5
feature_vector = np.array([dataset.features[sample_idx, :]])
cat_prob_data_by_method = construct_cat_prob_data_by_method(
    feature_vector, beta_samples_and_link_by_method
)

# ADD BMA
from categorical_from_binary.ib_cavi.cbm_vs_cbc.bma import (
    compute_weight_on_CBC_from_bayesian_model_averaging,
)


n_monte_carlo_samples = 10
ib_model = IB_Model.PROBIT  # TODO: Automatically extract this from the above.
CBC_weight = compute_weight_on_CBC_from_bayesian_model_averaging(
    covariates_train,
    labels_train,
    results.variational_params.beta,
    n_monte_carlo_samples,
    ib_model,
)
cat_prob_data_by_method = add_bma_to_cat_prob_data_by_method(
    cat_prob_data_by_method,
    CBC_weight,
    ib_model,
)


###
# Print out summary info about sampled category probabilities for a FIXED covariate vector
###

print("\n")
for method, cat_prob_data in cat_prob_data_by_method.items():
    print(
        f"{method} samples of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{cat_prob_data.samples}"
    )

print("\n")
for method, cat_prob_data in cat_prob_data_by_method.items():
    print(
        f"{method} means of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.mean(cat_prob_data.samples,0)}"
    )

print("\n")
for method, cat_prob_data in cat_prob_data_by_method.items():
    print(
        f"{method} stds of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.std(cat_prob_data.samples,0)}"
    )

print("\n")
for method, cat_prob_data in cat_prob_data_by_method.items():
    print(
        f"{method} 10th percentiles of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.percentile(cat_prob_data.samples,10,0)}"
    )

print("\n")
for method, cat_prob_data in cat_prob_data_by_method.items():
    print(
        f"{method} 90th percentiles of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.percentile(cat_prob_data.samples,90,0)}"
    )
