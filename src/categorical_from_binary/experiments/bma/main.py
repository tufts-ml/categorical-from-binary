"""
Explore the effectiveness of Bayesian Model Averaging in weighing the category probabilities
from IB+CBM and IB+CBC. 
    1) We use KL divergence from the estimates to the true (data generating)
        probabilities. 
    2) We plot the results over multiple datasets using dots. 
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.data_generation_configs import (
    get_num_parameters_from_num_covariates_and_num_categories,
    make_data_generation_configs,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.approximation_error import (
    compute_kl_divergence_from_estimated_to_true_category_probs,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.bma import (
    compute_weight_on_CBC_from_bayesian_model_averaging,
    construct_category_probabilities_from_bayesian_model_averaging,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    cbc_link_from_ib_model,
    cbm_link_from_ib_model,
    compute_ib_cavi_with_normal_prior,
)


###
# Configs
###

### Data generating configs
Ks = [3, 10]
multipliers_on_K_to_create_M = [1, 2]
multipliers_on_P_to_create_N = [10, 20, 40, 80, 160]
sigma_highs = [0.1, 2.0]
seed = 6
beta_0 = None
include_intercept = True
link = Link.MULTI_LOGIT
ib_model = IB_Model.LOGIT

### Inference configs
convergence_criterion_drop_in_mean_elbo = 0.1

### Plotting configs
sns_style = "darkgrid"
sns.set(style=sns_style)
save_filepath = f"data/results/bma/new_bma_plots_{ib_model.name}_{sns_style}"

###
# Code
###
data_generation_configs = make_data_generation_configs(
    Ks,
    multipliers_on_K_to_create_M,
    multipliers_on_P_to_create_N,
    sigma_highs,
    seed=seed,
)
print(f"The data generation configs are {data_generation_configs}")


results_dict = defaultdict(list)
for (s, dgc) in enumerate(data_generation_configs):
    print(f"----Now running simulation {s+1}/{len(data_generation_configs)}--")

    beta_category_strategy = ControlCategoryPredictability(
        scale_for_predictive_categories=dgc.scale_for_predictive_categories
    )
    dataset = generate_multiclass_regression_dataset(
        n_samples=dgc.n_samples,
        n_features=dgc.n_features,
        n_categories=dgc.n_categories,
        beta_0=beta_0,
        link=link,
        seed=dgc.seed,
        include_intercept=include_intercept,
        beta_category_strategy=beta_category_strategy,
    )

    # Prep training / test split
    n_train_samples = int(0.8 * dgc.n_samples)
    covariates_train = dataset.features[:n_train_samples]
    labels_train = dataset.labels[:n_train_samples]
    covariates_test = dataset.features[n_train_samples:]
    labels_test = dataset.labels[n_train_samples:]

    cbm_link = cbm_link_from_ib_model(ib_model)
    cbc_link = cbc_link_from_ib_model(ib_model)

    results = compute_ib_cavi_with_normal_prior(
        ib_model,
        labels_train,
        covariates_train,
        labels_test=labels_test,
        covariates_test=covariates_test,
        variational_params_init=None,
        convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
    )
    variational_beta = results.variational_params.beta

    # Approximate ELBO for each
    n_monte_carlo_samples = 10
    CBC_weight = compute_weight_on_CBC_from_bayesian_model_averaging(
        covariates_train,
        labels_train,
        variational_beta,
        n_monte_carlo_samples,
        ib_model,
    )
    print(f"For IB model {ib_model}, CBC weight is {CBC_weight}")

    ### check - does the weight make sense (compared to true!)
    probs_true = construct_category_probs(
        covariates_test, dataset.beta, link=Link.MULTI_LOGIT
    )
    probs_CBM = construct_category_probs(
        covariates_test,
        variational_beta.mean,
        link=cbm_link,
    )
    probs_CBC = construct_category_probs(
        covariates_test,
        variational_beta.mean,
        link=cbc_link,
    )
    probs_BMA = construct_category_probabilities_from_bayesian_model_averaging(
        covariates_test,
        variational_beta.mean,
        CBC_weight,
        ib_model,
    )

    divergence_from_CBM_to_true = (
        compute_kl_divergence_from_estimated_to_true_category_probs(
            probs_CBM, probs_true
        )
    )
    divergence_from_CBC_to_true = (
        compute_kl_divergence_from_estimated_to_true_category_probs(
            probs_CBC, probs_true
        )
    )
    divergence_from_BMA_to_true = (
        compute_kl_divergence_from_estimated_to_true_category_probs(
            probs_BMA, probs_true
        )
    )
    print(
        f"Mean divergence from CBM to true: {np.mean(divergence_from_CBM_to_true) }.  "
        f"Mean divergence from CBC to true: {np.mean(divergence_from_CBC_to_true)} "
        f"Mean divergence from BMA to true: {np.mean(divergence_from_BMA_to_true)}"
    )

    results_dict["N"].append(dgc.n_samples)
    results_dict["K"].append(dgc.n_categories)
    results_dict["M"].append(dgc.n_features)
    results_dict["sigma_high"].append(dgc.scale_for_predictive_categories)
    results_dict["ib_model"].append(str(ib_model))
    results_dict["CBC weight"].append(CBC_weight)
    results_dict["Mean KL divergence from CBM to true"].append(
        np.mean(divergence_from_CBM_to_true)
    )
    results_dict["Mean KL divergence from CBC to true"].append(
        np.mean(divergence_from_CBC_to_true)
    )
    results_dict["Mean KL divergence from BMA to true"].append(
        np.mean(divergence_from_BMA_to_true)
    )

df_results = pd.DataFrame(results_dict)
pd.set_option("display.float_format", lambda x: "%.6f" % x)
df_results

# If I want to convert the results dataframe to LateX, without the IB_Model.LOGIT column,
# print(df_results[df_results.columns.drop('ib_model')].to_latex(index=True, float_format="%.3f"))


df_results_melted = pd.melt(
    df_results,
    id_vars=["N", "K", "M", "sigma_high", "ib_model", "CBC weight"],
    var_name="Inference target",
    value_vars=[
        "Mean KL divergence from CBM to true",
        "Mean KL divergence from CBC to true",
        "Mean KL divergence from BMA to true",
    ],
    value_name="Mean KL divergence",
    ignore_index=False,
)
# make the index an actual column, with a name, so we can tell the plotter to use it on one of the axes
df_results_melted = df_results_melted.rename_axis("dataset_id").reset_index()
df_results_melted = df_results_melted.replace(
    "Mean KL divergence from CBM to true", "CBM"
)
df_results_melted = df_results_melted.replace(
    "Mean KL divergence from CBC to true", "CBC"
)
df_results_melted = df_results_melted.replace(
    "Mean KL divergence from BMA to true", "BMA"
)


####
#  Make plots showing asymptotic decrease in error for each context
###

# Reference for row and column headers:
# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
fig, axes = plt.subplots(4, 2, sharey=True, sharex=True, squeeze=True)
df_results = pd.DataFrame(results_dict)
contexts = []
for (s, sigma_high) in enumerate(sigma_highs):
    row = -1
    for K in Ks:
        for multiplier_on_K_to_create_M in multipliers_on_K_to_create_M:
            M = K * multiplier_on_K_to_create_M
            P = P = get_num_parameters_from_num_covariates_and_num_categories(K, M)
            df_results_context = df_results[
                (df_results["K"] == K)
                & (df_results["M"] == M)
                & (df_results["sigma_high"] == sigma_high)
            ]
            row += 1
            contexts.append(f"K={K}\n M={M}")
            n_multipliers_for_for_subplot = []
            kl_divs_from_BMA_for_subplot = []
            kl_divs_from_CBC_for_subplot = []
            kl_divs_from_CBM_for_subplot = []
            for (n, multiplier_on_P_to_create_N) in enumerate(
                multipliers_on_P_to_create_N
            ):
                N = P * multiplier_on_P_to_create_N
                n_multipliers_for_for_subplot.append(multiplier_on_P_to_create_N)
                kl_div_from_BMA = df_results_context.query(f"N == {N}")[
                    "Mean KL divergence from BMA to true"
                ].values[0]
                kl_div_from_CBM = df_results_context.query(f"N == {N}")[
                    "Mean KL divergence from CBM to true"
                ].values[0]
                kl_div_from_CBC = df_results_context.query(f"N == {N}")[
                    "Mean KL divergence from CBC to true"
                ].values[0]
                kl_divs_from_BMA_for_subplot.append(kl_div_from_BMA)
                kl_divs_from_CBM_for_subplot.append(kl_div_from_CBM)
                kl_divs_from_CBC_for_subplot.append(kl_div_from_CBC)
            axes[row, s].plot(
                n_multipliers_for_for_subplot,
                kl_divs_from_CBM_for_subplot,
                "-bo",
                label=f"CBM-{ib_model.name}",
                clip_on=False,
            )
            axes[row, s].plot(
                n_multipliers_for_for_subplot,
                kl_divs_from_CBC_for_subplot,
                "-r^",
                label=f"CBC-{ib_model.name}",
                clip_on=False,
            )
            axes[row, s].plot(
                n_multipliers_for_for_subplot,
                kl_divs_from_BMA_for_subplot,
                "-kx",
                label="BMA",
                clip_on=False,
                markersize=10,
            )
            axes[row, s].set_xticks(
                [n for n in n_multipliers_for_for_subplot],
                labels=[n for n in n_multipliers_for_for_subplot],
                size="x-small",
            )
            # axes[row,s].set_ylim(bottom=0)

fig.supylabel("Mean KL divergence to true probabilities")
fig.supxlabel("                    Ratio of sample size to number of parameters")

cols = [
    r"$\sigma_{high}$ = " + f"{sigma_highs[0]}",
    r"$\sigma_{high}$ = " + f"{sigma_highs[1]}",
]
rows = contexts

# plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label')

pad = 5  # in points

for ax, col in zip(axes[0], cols):
    ax.annotate(
        col,
        xy=(0.5, 1),
        xytext=(0, pad),
        xycoords="axes fraction",
        textcoords="offset points",
        size="medium",
        ha="center",
        va="baseline",
    )

for ax, row in zip(axes[:, 0], rows):
    ax.annotate(
        row,
        xy=(0, 0.5),
        xytext=(-ax.yaxis.labelpad - pad, 0),
        xycoords=ax.yaxis.label,
        textcoords="offset points",
        size="medium",
        ha="right",
        va="center",
    )


fig.subplots_adjust(left=0.25, top=0.75)
### Get labels from last axes and make legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(0.5, 0.95, 0.5, 0.075),
    fontsize="small",
)
# tight_layout doesn't take these labels into account. We'll need
# to make some room. These numbers are are manually tweaked.
# You could automatically calculate them, but it's a pain.
fig.tight_layout()
fig.savefig(save_filepath, bbox_inches="tight")
