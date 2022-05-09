"""
Train a categorical regresssion model for a SINGLE user in the cybersecurity dataset.
We can parallelize the training by using a slurm script. 
"""
import argparse
import os
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

from categorical_from_binary.autodiff.jax_helpers import optimize_beta_for_softmax_model
from categorical_from_binary.baserate import compute_probs_for_baserate_model
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    compute_covariate_conditional_entropies_of_true_category_probabilities,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.data_generation_configs import make_data_generation_configs
from categorical_from_binary.ib_cavi.cbm_vs_cbc.bma import (
    compute_weight_on_CBC_from_bayesian_model_averaging,
    construct_category_probabilities_from_bayesian_model_averaging,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_ib_cavi_with_normal_prior,
)
from categorical_from_binary.io import ensure_dir
from categorical_from_binary.metrics import (
    append_metrics_dict_for_one_dataset_to_results_dict,
    compute_metrics,
)
from categorical_from_binary.timing import time_me


def run_CAVI_vs_MLE_simulations(
    seed: int,
    list_of_n_categories: List[int],
    multipliers_on_n_categories_to_create_n_covariates: List[int],
    multipliers_on_n_parameters_to_create_n_samples: List[int],
    list_of_scales_for_predictive_categories: List[float],
    ib_model_as_string: str,
    data_generating_link_as_string: str,
    convergence_criterion_drop_in_mean_elbo: float,
    results_dir: str,
    test_run: bool = False,
    min_allowable_prob: Optional[float] = None,
    verbose: bool = True,
) -> None:
    """
    Runs simulations.  Generates one dataset per "data context" (K,M,N, ssq_high).
    Computes holdout likelhood for MLE and IB CAVI, as well as using baserate probs and true probs
    from the data generating process.

    Fixes the seed to some seed (so the whole suite of analyses can be distributed across HPC
    machines, with each one handling a different seed).

    Arguments:
        test_run: If True, we override the default results_dir and save things to a temp directory.

    Returns:
        None. Writes pandas dataframe to disk as csv.
    """
    # data generation configs that vary across data contexts
    data_generation_configs = make_data_generation_configs(
        list_of_n_categories,
        multipliers_on_n_categories_to_create_n_covariates,
        multipliers_on_n_parameters_to_create_n_samples,
        list_of_scales_for_predictive_categories,
        seed,
    )

    if test_run:
        results_dir = "./tmp"

    # items that are fixed across data contexts
    beta_0 = None
    include_intercept = True
    ib_model = IB_Model[ib_model_as_string.upper()]
    link_for_data_generation = Link[data_generating_link_as_string.upper()]

    results_dict = defaultdict(list)
    for (s, dgc) in enumerate(data_generation_configs):
        print(f"\n---Now running simulation with configs {dgc}---\n")
        beta_category_strategy = ControlCategoryPredictability(
            scale_for_predictive_categories=dgc.scale_for_predictive_categories
        )

        ###
        # Construct dataset and data metrics
        ###
        dataset = generate_multiclass_regression_dataset(
            n_samples=dgc.n_samples,
            n_features=dgc.n_features,
            n_categories=dgc.n_categories,
            beta_0=beta_0,
            link=link_for_data_generation,
            seed=seed,
            include_intercept=include_intercept,
            beta_category_strategy=beta_category_strategy,
        )

        # Prep training / test split
        n_train_samples = int(0.8 * dgc.n_samples)
        covariates_train = dataset.features[:n_train_samples]
        labels_train = dataset.labels[:n_train_samples]
        covariates_test = dataset.features[n_train_samples:]
        labels_test = dataset.labels[n_train_samples:]

        # compute entropy
        mean_covariate_conditional_entropy_of_true_category_probabilities = np.mean(
            (
                compute_covariate_conditional_entropies_of_true_category_probabilities(
                    dataset.features, dataset.beta, link_for_data_generation
                )
            )
        )
        if verbose:
            print(
                f"\nMean covariate conditional entropy of true category probs: {mean_covariate_conditional_entropy_of_true_category_probabilities}.\n"
            )

        ###
        # Inference
        ###

        # IB-CAVI
        results_CAVI, time_for_IB_CAVI = time_me(compute_ib_cavi_with_normal_prior)(
            ib_model,
            labels_train,
            covariates_train,
            labels_test=labels_test,
            covariates_test=covariates_test,
            variational_params_init=None,
            convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
        )
        variational_beta = results_CAVI.variational_params.beta
        pd.set_option("display.max_columns", None)
        if verbose:
            print(f"\n{results_CAVI.holdout_performance_over_time}")

        # MNL-MLE
        print("Now fitting softmax with MLE")
        beta_init = np.zeros((dgc.n_features + 1, dgc.n_categories))
        beta_star_softmax_MLE, time_for_softmax_MLE = time_me(
            optimize_beta_for_softmax_model
        )(
            beta_init,
            covariates_train,
            labels_train,
            verbose=verbose,
        )

        ###
        # Category probs
        ###
        probs_test_true = construct_category_probs(
            covariates_test,
            dataset.beta,
            link_for_data_generation,
        )

        probs_test_baserate = compute_probs_for_baserate_model(
            labels_train, len(labels_test)
        )
        probs_test_softmax_MLE = construct_category_probs(
            covariates_test,
            beta_star_softmax_MLE,
            Link.MULTI_LOGIT_NON_IDENTIFIED,
        )

        # Bayesian Model Averaging
        # TODO: Write a wrapper function around these two mini functions
        n_monte_carlo_samples = 10
        CBC_weight = compute_weight_on_CBC_from_bayesian_model_averaging(
            covariates_train,
            labels_train,
            variational_beta,
            n_monte_carlo_samples,
            ib_model,
        )
        probs_test_IB_CAVI_with_BMA = (
            construct_category_probabilities_from_bayesian_model_averaging(
                covariates_test,
                variational_beta.mean,
                CBC_weight,
                ib_model,
            )
        )

        ###
        # Metrics
        ###
        metrics_dict_for_one_dataset = {}
        metrics_dict_for_one_dataset["dgp"] = compute_metrics(
            probs_test_true, labels_test, min_allowable_prob
        )
        metrics_dict_for_one_dataset["baserate"] = compute_metrics(
            probs_test_baserate, labels_test, min_allowable_prob
        )
        metrics_dict_for_one_dataset["softmax_MLE"] = compute_metrics(
            probs_test_softmax_MLE, labels_test, min_allowable_prob
        )
        metrics_dict_for_one_dataset["IB_CAVI_plus_BMA"] = compute_metrics(
            probs_test_IB_CAVI_with_BMA, labels_test, min_allowable_prob
        )

        results_dict = append_metrics_dict_for_one_dataset_to_results_dict(
            metrics_dict_for_one_dataset, results_dict
        )

        ###
        # Update results dict
        ###
        results_dict["N"].append(dgc.n_samples)
        results_dict["K"].append(dgc.n_categories)
        results_dict["M"].append(dgc.n_features)
        results_dict["scale_for_predictive_categories"].append(
            dgc.scale_for_predictive_categories
        )
        results_dict["seed"].append(seed)
        results_dict["convergence_criterion_drop_in_mean_elbo"].append(
            convergence_criterion_drop_in_mean_elbo
        )
        results_dict["last_train_idx"].append(n_train_samples)
        results_dict[
            "mean_covariate_conditional_entropy_of_true_category_probabilities"
        ].append(mean_covariate_conditional_entropy_of_true_category_probabilities)
        # timing results
        results_dict["time for softmax MLE"].append(time_for_softmax_MLE)
        results_dict["time for IB CAVI"].append(time_for_IB_CAVI)

        ###
        # Print stuff
        ###
        if verbose:
            print(f"----Results----")
            for k, v in results_dict.items():
                print(f"{k} : {v[-1]:.03f}")

    ###
    # Save results as data frame
    ###
    ensure_dir(results_dir)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    path_to_results = os.path.join(
        results_dir,
        f"sim_results_CAVI_vs_MLE_seed={seed}_when_run={time_string}.csv",
    )
    results_dict_df = pd.DataFrame(results_dict)
    print(f"\nNow writing results to {path_to_results}.")
    results_dict_df.to_csv(path_to_results)


def _get_argument_parser():
    parser = argparse.ArgumentParser(description="Run CAVI vs MLE simulations")
    parser.add_argument(
        "--seed",
        type=int,
        help=f"The seed used for each dataset in the cartesian product of # categories, # covariates, # samples, ssq_high.",
    )
    parser.add_argument(
        "--list_of_n_categories",
        type=int,
        nargs="+",
        default=[3],
        help=f"List of integers reflecting the number of categories we want for each run.",
    )
    parser.add_argument(
        "--multipliers_on_n_categories_to_create_n_covariates",
        type=int,
        nargs="+",
        default=[1],
        help=f"List of integers reflecting multipliers on # categories to create # covariates.",
    )
    parser.add_argument(
        "--multipliers_on_n_parameters_to_create_n_samples",
        type=int,
        nargs="+",
        default=[10],
        help=f"List of integers reflecting multipliers on # parameters to create # samples.",
    )
    parser.add_argument(
        "--list_of_scales_for_predictive_categories",
        type=float,
        nargs="+",
        default=[0.1],
        help=f"List of floats reflecting the value of ssq_high when generating data."
        f"Higher values make the responses more predictable from the covariates",
    )

    parser.add_argument(
        "--ib_model_as_string",
        type=str,
        default="probit",
        help=f"The type of IB model we want to fit (which describes H^inv, the link function).  One of: probit, logit.",
    )
    parser.add_argument(
        "--data_generating_link_as_string",
        type=str,
        default="multi_logit",
        help=f"Link for generating data.  Needs to be a member of the Link Enum.",
    )
    parser.add_argument(
        "--convergence_criterion_drop_in_mean_elbo",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/cluster/tufts/hugheslab/mwojno01/data/results/sims/no_dir_specified/",
        help=f"directory for saving results of the run (assuming we're not running a test)",
    )
    parser.add_argument(
        "--test_run",
        type=bool,
        default=False,
        help=f"If this flag is present, we override the default results_dir and save things to a temp directory",
    )
    return parser


if __name__ == "__main__":
    """
    Usage:
        # Test run 
            python src/categorical_from_binary/experiments/sims/main/run_sims.py  --seed 0 --test_run true 
        # More elaborate test run 
            python src/categorical_from_binary/experiments/sims/main/run_sims.py --seed 0 \
                --list_of_n_categories 3 10 \
                --multipliers_on_n_categories_to_create_n_covariates 1 2 \
                --multipliers_on_n_parameters_to_create_n_samples 10 100 \
                --list_of_scales_for_predictive_categories  0.1 2.0 \
                --ib_model_as_string "probit" \
                --data_generating_link_as_string "multi_logit" \
                --convergence_criterion_drop_in_mean_elbo 0.1 \
                --results_dir "./tmp" 
        # For real run, do the above but with the right parameters. If on the cluster, can use the default results_dir
    """

    args = _get_argument_parser().parse_args()
    seed = args.seed
    list_of_n_categories = args.list_of_n_categories
    multipliers_on_n_categories_to_create_n_covariates = (
        args.multipliers_on_n_categories_to_create_n_covariates
    )
    multipliers_on_n_parameters_to_create_n_samples = (
        args.multipliers_on_n_parameters_to_create_n_samples
    )
    list_of_scales_for_predictive_categories = (
        args.list_of_scales_for_predictive_categories
    )
    ib_model_as_string = args.ib_model_as_string
    data_generating_link_as_string = args.data_generating_link_as_string
    convergence_criterion_drop_in_mean_elbo = (
        args.convergence_criterion_drop_in_mean_elbo
    )
    results_dir = args.results_dir
    test_run = args.test_run
    run_CAVI_vs_MLE_simulations(
        seed,
        list_of_n_categories,
        multipliers_on_n_categories_to_create_n_covariates,
        multipliers_on_n_parameters_to_create_n_samples,
        list_of_scales_for_predictive_categories,
        ib_model_as_string,
        data_generating_link_as_string,
        convergence_criterion_drop_in_mean_elbo,
        results_dir,
        test_run,
    )
