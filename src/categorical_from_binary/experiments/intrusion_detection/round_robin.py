"""
This script takes as input ONE user's model (just the posterior mean) from training, 
and uses it to score data from all the users for which there were models.

We will eventually plot the round robin results in heatmap (self vs other) form.
"""


import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import sparse  # noqa

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.datasets.cyber.load import (
    construct_process_start_features_and_labels_for_one_cyber_user,
)
from categorical_from_binary.io import ensure_dir
from categorical_from_binary.metrics import compute_metrics
from categorical_from_binary.performance_over_time.configs_util import Configs
from categorical_from_binary.performance_over_time.main import (
    load_configs,
    update_configs_via_optional_overrides,
)


def _get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run performance over time from path to yaml configs"
    )
    parser.add_argument(
        "--path_to_configs",
        type=str,
    )
    parser.add_argument(
        "--cyber_user_idx_relative_to_subset_override",
        type=int,
        help=f"Can be used to override user idx relative to subset when operating on cyber data",
        default=None,
    )
    parser.add_argument(
        "--scoring_results_dir",
        type=str,
        help=f"Directory to where we write scoring results",
    )
    parser.add_argument(
        "--cavi_time_units",
        type=int,
        help=f"Determines how long training was for IB-CAVI before grabbing betas",
    )
    parser.add_argument(
        "--advi_time_units",
        type=int,
        help=f"Determines how long training was for ADVI before grabbing betas",
    )

    return parser


def load_beta_mean_and_get_user_domain_for_cyber_user(
    user_idx: int,
    cavi_or_advi_string: str,
    cavi_time_units: int,
    advi_time_units: int,
    training_results_master_dir: str = "/cluster/tufts/hugheslab/mwojno01/data/intrusion/",
) -> Tuple[np.array, str]:
    """
    This function makes a VERY specific assumption that the betas were stored at locations with a particular structure.
    This structure is currently imposed by the performance over time training function.


    Example path:
        "/cluster/tufts/hugheslab/mwojno01/data/intrusion/user_300/U137@C808/05_19_2022_18_31_26_MDT_advi/betas/ADVI/lr=1.0/save_every_secs=1200.0_units_of_save_every=8_secs_elapsed=10486.810/beta_mean.npy"
    """
    user_dir = f"{training_results_master_dir}user_{user_idx}"
    user_domain = next(os.walk(user_dir))[1][0]
    user_dir_with_user_domain = f"{user_dir}/{user_domain}/"
    date_method = [
        date_method
        for date_method in next(os.walk(user_dir_with_user_domain))[1]
        if cavi_or_advi_string in date_method
    ][0]
    user_beta_master_dir = f"{user_dir_with_user_domain}/{date_method}/betas/"
    inference_type_with_hyperparam = (
        "IB_CAVI"
        if cavi_or_advi_string == "cavi"
        else f"{cavi_or_advi_string.upper()}/lr=1.0/"
    )
    user_beta_dir_with_inference = (
        f"{user_beta_master_dir}/{inference_type_with_hyperparam}/"
    )
    time_units = cavi_time_units if cavi_or_advi_string == "cavi" else advi_time_units
    beta_snapshot = [
        beta_snap
        for beta_snap in next(os.walk(user_beta_dir_with_inference))[1]
        if f"units_of_save_every={time_units}" in beta_snap
    ][0]
    total_child_dir = f"{user_beta_dir_with_inference}/{beta_snapshot}/"
    beta_mean_basename = [
        basename
        for basename in next(os.walk(total_child_dir))[2]
        if f"beta_mean" in basename
    ][0]

    beta_mean_path = os.path.join(total_child_dir, beta_mean_basename)
    if ".npz" in beta_mean_path:
        beta_mean = scipy.sparse.load_npz(beta_mean_path)
    elif ".npy" in beta_mean_path:
        beta_mean = np.load(beta_mean_path)  # not sure if this works.
    else:
        raise ValueError("I don't know how to load betas.")
    return beta_mean, user_domain


# TODO: for now, the `configs` are all for training, and the other stuff just gets entered at command line or by default
# Consider changing to make scoring configs (would probably be better).


def score_all_data_with_one_users_model_from_loaded_configs(
    configs: Configs,
    user_idx_relative_to_subset: int,
    scoring_results_dir: str,
    cavi_time_units: int,
    advi_time_units: int,
):

    mean_log_likes = []
    data_user_domains = []

    for cavi_or_advi_string in ["cavi", "advi"]:

        ### Load model for model_user_idx
        model_user_idx = (
            configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events
            + user_idx_relative_to_subset
        )
        (
            beta_mean,
            model_user_domain,
        ) = load_beta_mean_and_get_user_domain_for_cyber_user(
            model_user_idx, cavi_or_advi_string, cavi_time_units, advi_time_units
        )

        ### Get TEST data to score from many users (self and others)
        for user_idx_relative_to_subset in range(
            configs.data.cyber.subset_number_of_users
        ):
            (
                covariates,
                labels,
                data_user_domain,
            ) = construct_process_start_features_and_labels_for_one_cyber_user(
                configs.data.cyber.path_to_human_process_start_data,
                configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events,
                configs.data.cyber.subset_number_of_users,
                user_idx_relative_to_subset,
                configs.data.cyber.window_size,  # TODO: should be obtained from file / metadata
                configs.data.cyber.temperature,  # TODO: should be obtained from file /metadata
                configs.data.cyber.include_intercept,  # TODO: should be obtained from file / metadata
            )  # noqa

            data_user_domains.append(data_user_domain)

            # Grab TEST data!
            n_train_samples = int(
                configs.data.cyber.pct_training * np.shape(covariates)[0]
            )
            covariates_test = covariates[n_train_samples:]
            labels_test = labels[n_train_samples:]

            ### Scoring
            try:
                # TODO: Add BMA instead of assuming that PROBIT is the right model.
                link = (
                    Link.CBC_PROBIT
                    if cavi_or_advi_string == "cavi"
                    else Link(configs.holdout_performance.advi.link)
                )
                probs_test = construct_category_probs(
                    covariates_test,
                    beta_mean,
                    link,
                )
                metrics = compute_metrics(probs_test, labels_test)
                mean_log_like = metrics.mean_log_like
                print(
                    f"\t The mean log like for {model_user_domain} scoring {data_user_domain} was {mean_log_like}"
                )

                mean_log_likes.append(mean_log_like)
            except:  # some users don't have data for some reason
                print(
                    f"Could not process model_user {model_user_domain}, data user {data_user_domain}"
                )
                mean_log_likes.append(np.nan)

        # save results
        ensure_dir(scoring_results_dir)
        scoring_results_df_path = os.path.join(
            scoring_results_dir,
            f"{cavi_or_advi_string}_model_user={model_user_domain}_model_user_idx_={model_user_idx}_mean_log_likes_round_robin.csv",
        )
        df_metrics = pd.DataFrame(mean_log_likes, index=data_user_domains)
        df_metrics.to_csv(scoring_results_df_path)


def score_all_data_with_one_users_model(
    path_to_configs: str,
    cyber_user_idx_relative_to_subset_override,
    scoring_results_dir,
    cavi_time_units,
    advi_time_units,
) -> None:
    """
    Score all data with one users model from yaml configs, but allow overrides (so we can control which user model
    we are scoring with from command line, which allows for easy parallelization on the cluster)
    """

    configs = load_configs(path_to_configs)
    configs = update_configs_via_optional_overrides(
        configs,
        cyber_user_idx_relative_to_subset_override=cyber_user_idx_relative_to_subset_override,
    )
    score_all_data_with_one_users_model_from_loaded_configs(
        configs,
        cyber_user_idx_relative_to_subset_override,
        scoring_results_dir,
        cavi_time_units,
        advi_time_units,
    )


if __name__ == "__main__":
    """
    Usage:
        Example:
            python src/categorical_from_binary/performance_over_time/main.py  --path_to_configs configs/performance_over_time/demo_sims.yaml --only_run_this_inference 1
    """

    args = _get_argument_parser().parse_args()
    score_all_data_with_one_users_model(
        args.path_to_configs,
        args.cyber_user_idx_relative_to_subset_override,
        args.scoring_results_dir,
        args.cavi_time_units,
        args.advi_time_units,
    )
