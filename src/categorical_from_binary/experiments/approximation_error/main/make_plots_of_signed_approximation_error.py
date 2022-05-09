"""
Compare IB+CBM vs IB+CBC 
by plotting signed approximation error against
dominant category probability.  This module specifically
investigates the approximation error when using the *MLEs*
for the IB model. 

This code will create multiple datasets with multiple
configs and then 
"""

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


np.set_printoptions(precision=3, suppress=True)

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions,
    compute_CBM_Probit_predictions,
    compute_IB_Probit_predictions,
    optimize_beta_for_CBC_model,
    optimize_beta_for_CBM_model,
    optimize_beta_for_IB_model,
)
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.approximation_error import (
    compute_signed_error_in_largest_category_probs,
)
from categorical_from_binary.io import ensure_dir
from categorical_from_binary.plot_helpers import make_linear_regression_plot


@dataclass
class DataGenerationConfigs:
    seed: int
    n_categories: int
    n_features: int
    n_samples: int
    scale_for_predictive_categories: float


# set configs related to data generation

data_generation_configs = [
    DataGenerationConfigs(4, 3, 1, 1800, 0.1),
    DataGenerationConfigs(6, 20, 40, 12000, 0.1),
]


# data_generation_configs=[
#     DataGenerationConfigs(1, 3, 6, 1800, 2.0),
#     DataGenerationConfigs(2, 10, 20, 6000, 2.0),
#     DataGenerationConfigs(3, 20, 40, 12000, 2.0),
#     DataGenerationConfigs(4, 3, 6, 1800, 0.1),
#     DataGenerationConfigs(5, 10, 20, 6000, 0.1),
#     DataGenerationConfigs(6, 20, 40, 12000, 0.1),
# ]


beta_0 = None
include_intercept = True
link = Link.MULTI_LOGIT
dict_of_error_before_and_after_averaging = defaultdict(list)
results_by_dataset = []

# set configs relative to saving data
SAVE_DIR = "data/results/evaluating_IB_approximation_targets/"
ensure_dir(SAVE_DIR)

for dgc in data_generation_configs:
    ###
    # Construct dataset
    ###
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

    ###
    # Compute MLE's
    ###

    beta_init = np.zeros((dgc.n_features + 1, dgc.n_categories))

    print("Now doing MLE with IB model")
    beta_star_IB_MLE = optimize_beta_for_IB_model(
        beta_init, dataset.features, dataset.labels
    )

    print("Now doing MLE with CBC model")
    beta_star_CBC_MLE = optimize_beta_for_CBC_model(
        beta_init, dataset.features, dataset.labels
    )

    print("Now doing MLE with CBM model")
    beta_star_CBM_MLE = optimize_beta_for_CBM_model(
        beta_init, dataset.features, dataset.labels
    )

    ###
    # Compute predictions
    ###

    CBC_predictions_with_IB_MLE = compute_CBC_Probit_predictions(
        beta_star_IB_MLE, dataset.features, return_numpy=True
    )
    CBC_predictions_with_CBC_MLE = compute_CBC_Probit_predictions(
        beta_star_CBC_MLE, dataset.features, return_numpy=True
    )

    CBM_predictions_with_IB_MLE = compute_CBM_Probit_predictions(
        beta_star_IB_MLE, dataset.features, return_numpy=True
    )
    CBM_predictions_with_CBM_MLE = compute_CBM_Probit_predictions(
        beta_star_CBM_MLE, dataset.features, return_numpy=True
    )

    IB_predictions_with_IB_MLE = compute_IB_Probit_predictions(
        beta_star_IB_MLE, dataset.features
    )
    true_data_generating_probabilities = construct_category_probs(
        dataset.features, dataset.beta, link
    )

    predicted_categories_IB_plus_CBC = np.argmax(CBC_predictions_with_IB_MLE, 1)
    largest_probs_IB_plus_CBC = np.array(
        [
            CBC_predictions_with_IB_MLE[i, k]
            for (i, k) in enumerate(predicted_categories_IB_plus_CBC)
        ]
    )
    signed_errors_IB_plus_CBC = compute_signed_error_in_largest_category_probs(
        CBC_predictions_with_IB_MLE, CBC_predictions_with_CBC_MLE
    )

    predicted_categories_IB_plus_CBM = np.argmax(CBM_predictions_with_IB_MLE, 1)
    largest_probs_IB_plus_CBM = np.array(
        [
            CBM_predictions_with_IB_MLE[i, k]
            for (i, k) in enumerate(predicted_categories_IB_plus_CBM)
        ]
    )
    signed_errors_IB_plus_CBM = compute_signed_error_in_largest_category_probs(
        CBM_predictions_with_IB_MLE, CBM_predictions_with_CBM_MLE
    )

    df_error_direction = pd.DataFrame(
        {
            "largest_probs_IB_plus_CBC": largest_probs_IB_plus_CBC,
            "signed_errors_IB_plus_CBC": signed_errors_IB_plus_CBC,
            "largest_probs_IB_plus_CBM": largest_probs_IB_plus_CBM,
            "signed_errors_IB_plus_CBM": signed_errors_IB_plus_CBM,
        }
    )

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, sharey=True)
    make_linear_regression_plot(
        df_error_direction,
        x_col_name="largest_probs_IB_plus_CBC",
        y_col_name="signed_errors_IB_plus_CBC",
        x_axis_label="largest category probability \n (CBC-Probit with IB-Probit MLE)",
        y_axis_label="signed error to exact inference \n (CBC-Probit with CBC-Probit MLE)",
        title="",
        add_y_equals_x_line=False,
        add_y_equals_0_line=True,
        lowess=True,
        plot_points_as_density_not_scatter=True,
        ax=axes[0],
    )
    make_linear_regression_plot(
        df_error_direction,
        x_col_name="largest_probs_IB_plus_CBM",
        y_col_name="signed_errors_IB_plus_CBM",
        x_axis_label="largest category probability \n (CBM-Probit with IB-Probit MLE)",
        y_axis_label="signed error to exact inference \n (CBM-Probit with CBM-Probit MLE)",
        title="",
        add_y_equals_x_line=False,
        add_y_equals_0_line=True,
        lowess=True,
        plot_points_as_density_not_scatter=True,
        ax=axes[1],
    )
    plt.tight_layout(pad=1.0)

    y_vals = list(df_error_direction["signed_errors_IB_plus_CBM"]) + list(
        df_error_direction["signed_errors_IB_plus_CBC"]
    )
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    basename = f"directional_approximation_error_seed={dgc.seed}_K={dgc.n_categories}_M={dgc.n_features}_N={dgc.n_samples}_ssqhigh={dgc.scale_for_predictive_categories}.png"
    pathname = os.path.join(SAVE_DIR, basename)
    plt.savefig(pathname)
    # plt.show()

    # create_hot_pairplot(df_error_direction)
    # plt.show()
