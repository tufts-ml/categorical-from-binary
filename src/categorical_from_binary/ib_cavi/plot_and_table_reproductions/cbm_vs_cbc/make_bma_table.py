"""
Explore the effectiveness of Bayesian Model Averaging in weighing the category probabilities
from IB+CBM and IB+CBC.

This made a table which currently lives in the cbm_vs_cbc report (in the independent binary
approximations repo.)
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.approximation_error import (
    compute_approximation_error_in_category_probs_using_l1_distance,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.bma import (
    compute_weight_on_CBC_from_bayesian_model_averaging,
    construct_category_probabilities_from_bayesian_model_averaging,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_multiclass_vi_with_normal_prior,
    do_link_from_ib_model,
    sdo_link_from_ib_model,
)


@dataclass
class DataGenerationConfigs:
    seed: int
    n_categories: int
    n_features: int
    n_samples: int
    scale_for_predictive_categories: float


data_generation_configs = [
    DataGenerationConfigs(1, 3, 1, 100, 2.0),
    DataGenerationConfigs(2, 3, 1, 100, 2.0),
    DataGenerationConfigs(3, 3, 1, 100, 2.0),
    DataGenerationConfigs(1, 3, 1, 1800, 2.0),
    DataGenerationConfigs(2, 3, 1, 1800, 2.0),
    DataGenerationConfigs(3, 3, 1, 1800, 2.0),
    DataGenerationConfigs(1, 3, 6, 1800, 0.1),
    DataGenerationConfigs(2, 3, 6, 1800, 0.1),
    DataGenerationConfigs(3, 3, 6, 1800, 0.1),
    DataGenerationConfigs(1, 20, 40, 12000, 0.1),
    DataGenerationConfigs(2, 20, 40, 12000, 0.1),
    DataGenerationConfigs(3, 20, 40, 12000, 0.1),
    DataGenerationConfigs(1, 20, 40, 12000, 2.0),
    DataGenerationConfigs(2, 20, 40, 12000, 2.0),
    DataGenerationConfigs(3, 20, 40, 12000, 2.0),
]

# data generating configs that are constant across run
beta_0 = None
include_intercept = True
link = Link.MULTI_LOGIT
ib_models = [IB_Model.LOGIT, IB_Model.PROBIT]
results_dict = defaultdict(list)
convergence_criterion_drop_in_mean_elbo = 0.1

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

    for ib_model in ib_models:
        sdo_link = sdo_link_from_ib_model(ib_model)
        do_link = do_link_from_ib_model(ib_model)

        results = compute_multiclass_vi_with_normal_prior(
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
            link=sdo_link,
        )
        probs_CBC = construct_category_probs(
            covariates_test,
            variational_beta.mean,
            link=do_link,
        )
        probs_BMA = construct_category_probabilities_from_bayesian_model_averaging(
            covariates_test,
            variational_beta.mean,
            CBC_weight,
            ib_model,
        )

        error_CBM_to_true = (
            compute_approximation_error_in_category_probs_using_l1_distance(
                probs_CBM, probs_true
            )
        )
        error_CBC_to_true = (
            compute_approximation_error_in_category_probs_using_l1_distance(
                probs_CBC, probs_true
            )
        )
        error_BMA_to_true = (
            compute_approximation_error_in_category_probs_using_l1_distance(
                probs_BMA, probs_true
            )
        )
        print(
            f"Mean error CBM to true: {np.mean(error_CBM_to_true) }.  "
            f"Mean error CBC to true: {np.mean(error_CBC_to_true)} "
            f"Mean error BMA to true: {np.mean(error_BMA_to_true)}"
        )

        results_dict["N"].append(dgc.n_samples)
        results_dict["K"].append(dgc.n_categories)
        results_dict["M"].append(dgc.n_features)
        results_dict["ssq_high"].append(dgc.scale_for_predictive_categories)
        results_dict["ib_model"].append(str(ib_model))
        results_dict["CBC weight"].append(CBC_weight)
        results_dict["Mean L1 holdout error CBM to true"].append(
            np.mean(error_CBM_to_true)
        )
        results_dict["Mean L1 holdout error CBC to true"].append(
            np.mean(error_CBC_to_true)
        )
        results_dict["Mean L1 holdout error BMA to true"].append(
            np.mean(error_BMA_to_true)
        )

results_df = pd.DataFrame(results_dict)
pd.set_option("display.float_format", lambda x: "%.6f" % x)
results_df
# print(results_df.to_latex(float_format="%.4f"))
