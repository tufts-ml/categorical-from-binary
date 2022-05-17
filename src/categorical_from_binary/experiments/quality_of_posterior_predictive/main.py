import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.experiments.quality_of_posterior_predictive.helpers import (
    BetaSamplesAndLink,
    make_df_of_sampled_category_probs_for_each_method_and_covariate_vector,
    sample_beta_advi,
    sample_beta_cavi,
)
from categorical_from_binary.hmc.core import (
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.bma import (
    compute_weight_on_CBC_from_bayesian_model_averaging,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.multi.inference import IB_Model
from categorical_from_binary.io import ensure_dir
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
n_categories = 4
n_sparse_categories = 0
n_features = 8
n_samples = 1000
n_train_samples = int(0.8 * n_samples)
include_intercept = True
link_true_model = Link.MULTI_LOGIT
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
    link=link_true_model,
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


# Get weights for BMA
n_monte_carlo_samples = 10
ib_model = IB_Model.PROBIT  # TODO: Automatically extract this from the above.
CBC_weight = compute_weight_on_CBC_from_bayesian_model_averaging(
    covariates_train,
    labels_train,
    results.variational_params.beta,
    n_monte_carlo_samples,
    ib_model,
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
    name_for_nuts = f"{link_for_nuts.name}+NUTS"
    beta_samples_and_link_by_method[name_for_nuts] = BetaSamplesAndLink(
        beta_samples_for_nuts, link_for_nuts
    )

###
# ADVI Inference
###
link_advi = Link.CBC_PROBIT
lr = 0.1
n_advi_iterations = 1000
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
    link_advi,
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
    beta_samples_and_link_by_method[f"{link_cavi.name}+IB-CAVI"] = BetaSamplesAndLink(
        beta_samples_cavi, link_cavi
    )


# ADVI
M, L = np.shape(beta_mean_ADVI_CBC)
beta_samples_advi = np.zeros((num_mcmc_samples, M, L))
for i in range(num_mcmc_samples):
    beta_samples_advi[i, :] = sample_beta_advi(
        beta_mean_ADVI_CBC, beta_stds_ADVI_CBC, seed=i
    )
beta_samples_and_link_by_method[f"{link_advi.name}+ADVI"] = BetaSamplesAndLink(
    beta_samples_advi, link_advi
)


###
#  (for plotting posterior dist'n for category probabilities over some examples)
###

# Reference: https://seaborn.pydata.org/examples/grouped_violinplots.html

# configs
save_dir = "/Users/mwojno01/Desktop/cat_prob_dist_plots_fixed_names/"
example_idxs = np.arange(0, 10)
covariates = covariates_train
colors_by_methods_to_plot = {
    "CBC_PROBIT+ADVI": "b",
    "CBC_PROBIT+NUTS": "g",
    "CBC_PROBIT+IB-CAVI": "r",
    #    'ib_cavi_CBM_PROBIT':"orange"
}

# dataframe
df = make_df_of_sampled_category_probs_for_each_method_and_covariate_vector(
    covariates,
    CBC_weight,
    beta_samples_and_link_by_method,
    example_idxs,
    colors_by_methods_to_plot,
    n_categories,
    num_mcmc_samples,
    ib_model,
)


# data generating process
cat_probs_dgp = construct_category_probs(covariates, dataset.beta, link=link_true_model)


# plot
ensure_dir(save_dir)
for show_only_plot_rather_than_only_legend in [False, True]:
    for example_idx in example_idxs:
        print(f"Now saving plot for example {example_idx}")
        plt.clf()
        sns.set_theme(style="whitegrid")
        sns.set(font_scale=3)
        df_single = df.query(f"example=={example_idx}")
        g = sns.violinplot(
            data=df_single,
            x="category",
            y="prob",
            hue="method",
            split=False,
            inner="quart",
            linewidth=1,
            palette=colors_by_methods_to_plot,
        )
        sns.despine(left=True)
        EPSILON = 0.1
        g.set(ylim=(0 - EPSILON, 1 + EPSILON))
        g.set(ylabel=None, xlabel=None)
        # add ground truth
        cat_probs_true = cat_probs_dgp[example_idx]

        for k in range(n_categories):
            if k == 0:
                # TODO: Don't hardcode
                label = f"True Model (SOFTMAX)"
            else:
                label = None
            plt.hlines(
                y=cat_probs_true[k],
                xmin=k - (1 / n_categories),
                xmax=k + (1 / n_categories),
                colors="k",
                linestyles="--",
                label=label,
            )

        if show_only_plot_rather_than_only_legend:
            # remove legend
            plt.legend([], [], frameon=False)
            basename = f"cat_prob_plots_K={n_categories}_M={n_features}_N={n_samples}_example_idx={example_idx}.png"
            filepath = os.path.join(save_dir, basename)
            plt.tight_layout()
            plt.savefig(filepath, dpi="figure", bbox_inches="tight")
        else:
            legend = plt.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0
            )
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()
            )
            # bbox  = legend.get_window_extent()
            # expand=[-5,-5,5,5]
            # bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            # bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            # show only legend; remove plot.
            basename = f"cat_prob_plots_K={n_categories}_M={n_features}_N={n_samples}_legend_only.png"
            filepath = os.path.join(save_dir, basename)
            plt.tight_layout()
            plt.savefig(filepath, dpi="figure", bbox_inches=bbox)
            break


# ###
# # Get category probability samples for a FIXED covariate vector
# ###

# sample_idx = 4
# feature_vector = np.array([dataset.features[sample_idx, :]])
# cat_prob_data_by_method = construct_cat_prob_data_by_method(
#     feature_vector, beta_samples_and_link_by_method
# )

# # ADD BMA
# cat_prob_data_by_method = add_bma_to_cat_prob_data_by_method(
#     cat_prob_data_by_method,
#     CBC_weight,
#     ib_model,
# )


# ###
# # Print out summary info about sampled category probabilities for a FIXED covariate vector
# ###

# print("\n")
# for method, cat_prob_data in cat_prob_data_by_method.items():
#     print(
#         f"{method} samples of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{cat_prob_data.samples}"
#     )

# print("\n")
# for method, cat_prob_data in cat_prob_data_by_method.items():
#     print(
#         f"{method} means of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.mean(cat_prob_data.samples,0)}"
#     )

# print("\n")
# for method, cat_prob_data in cat_prob_data_by_method.items():
#     print(
#         f"{method} stds of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.std(cat_prob_data.samples,0)}"
#     )

# print("\n")
# for method, cat_prob_data in cat_prob_data_by_method.items():
#     print(
#         f"{method} 10th percentiles of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.percentile(cat_prob_data.samples,10,0)}"
#     )

# print("\n")
# for method, cat_prob_data in cat_prob_data_by_method.items():
#     print(
#         f"{method} 90th percentiles of category probabilities (link={cat_prob_data.link.name}) for a single holdout observation: \n{np.percentile(cat_prob_data.samples,90,0)}"
#     )
