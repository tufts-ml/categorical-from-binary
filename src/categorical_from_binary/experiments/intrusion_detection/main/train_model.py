"""
Train a categorical regresssion model for a SINGLE user in the cybersecurity dataset.
We can parallelize the training by using a slurm script. 
"""
import argparse
import faulthandler
import os

from categorical_from_binary.datasets.cyber.data_frame import (
    load_human_process_start_df,
)
from categorical_from_binary.datasets.cyber.featurize import (
    construct_features,
    construct_labels,
)
from categorical_from_binary.datasets.cyber.util import (
    compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id,
    compute_sorted_sample_sizes_for_users,
    construct_minimal_process_id_by_process_id,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.io import ensure_dir, write_json
from categorical_from_binary.pandas_helpers import keep_df_rows_by_column_values
from categorical_from_binary.performance_over_time.vi_params import (
    VI_results_from_CAVI_results,
    write_VI_results,
)


def _get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train a user-specific process start model"
    )
    parser.add_argument(
        "--user_idx_relative_to_subset",
        type=int,
        help=f"index of user-domain within human process start dataframe after sorting (in descending order) by size."
        f" Must be an integer in [0,number_of_users_in_subset]",
    )
    parser.add_argument(
        "--number_of_users_in_subset",
        type=int,
        help=f"The number of users in the collection from which we determine the number of categories "
        f"(number of process starts) to use for analysis. "
        f" Must be an integer in [0,10619]",
    )
    parser.add_argument(
        "--start_idx_for_users_in_subset",
        type=int,
        help=f"First index at which we start processing the data in order to determine "
        f"the number of categories (number of process starts).  The users are "
        f"sorted by the number of (process start) events, so the lower the value for this, the more observations "
        f"that there will be per user."
        f" Must be an integer in [0,10619]",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help=f"number of processes in lookback window used for featurization",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=60.0,
        help=f"temperature (in seconds); higher values means older process starts will be more influential in the regression ",
    )
    parser.add_argument(
        "--max_n_iterations",
        type=int,
        default=20,
        help=f"number of iterations used for CAVI",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/cluster/tufts/hugheslab/mwojno01/data/results/",
        help=f"directory for saving results of the run (assuming we're not running a test)",
    )
    parser.add_argument(
        "--path_to_human_process_start_data",
        type=str,
        default="/cluster/tufts/hugheslab/mwojno01/data/human_process_starts.csv",
        help=f"directory containing human process start data",
    )
    parser.add_argument(
        "--test_run",
        type=bool,
        default=False,
        help=f"If this flag is present, a minimal version is run as a test to make sure everything works ok",
    )
    return parser


def train_model(
    user_idx_relative_to_subset: int,
    number_of_users_in_subset: int,
    start_idx_for_users_in_subset: int,
    window_size: int,
    temperature: float,
    max_n_iterations: int,
    path_to_human_process_start_data: str,
    save_dir: str,
    test_run: bool,
):
    """
    Arguments:
        These are explained in the help fields for the argparser above.
    """
    print(
        f" Now training IB-CAVI on cybersecurity data for user idx {user_idx_relative_to_subset}, "
        f" number_of_users_in_subset {number_of_users_in_subset}, start_idx_for_users_in_subset {start_idx_for_users_in_subset} "
        f" # iterations {max_n_iterations}, "
        f" window size {window_size}, temperature {temperature}, and save directory {save_dir}.  Is this a test run? {test_run} "
    )

    faulthandler.enable()

    if test_run:
        save_dir = "./tmp"

    df_all = load_human_process_start_df(path_to_human_process_start_data)

    # get a subset of all user domains, from which we determine the number of categories to use for processing.
    sorted_sample_sizes_for_users = compute_sorted_sample_sizes_for_users(df_all)
    user_domains_in_subset = sorted_sample_sizes_for_users.index[
        start_idx_for_users_in_subset : start_idx_for_users_in_subset
        + number_of_users_in_subset
    ]
    df_subset = keep_df_rows_by_column_values(
        df_all, col="user@domain", values=user_domains_in_subset
    )
    minimal_process_id_by_process_id = construct_minimal_process_id_by_process_id(
        df_subset
    )
    K = compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
        minimal_process_id_by_process_id
    )
    print(
        f"The total number of distinct processes started in this subset of users is {K}."
    )
    # `K` is the number of unique processes

    ### Get data for one user
    sorted_sample_sizes_for_users = compute_sorted_sample_sizes_for_users(df_all)
    user_domain = user_domains_in_subset[user_idx_relative_to_subset]
    df = keep_df_rows_by_column_values(df_all, col="user@domain", values=user_domain)

    if test_run:
        df = df[:100]

    ###
    # Try to free memory
    ###
    del df_all
    import gc

    gc.collect()

    ###
    # Featurize the dataset
    ###
    labels = construct_labels(df, minimal_process_id_by_process_id, window_size)
    features = construct_features(
        df,
        minimal_process_id_by_process_id,
        window_size,
        temperature,
        include_intercept=True,
    )

    ####
    # Run inference
    ####

    # Prep training / test split
    N = features.shape[0]
    n_train_samples = int(0.8 * N)
    covariates_train = features[:n_train_samples]
    labels_train = labels[:n_train_samples]
    covariates_test = features[n_train_samples:]
    labels_test = labels[n_train_samples:]

    print(f"About to run CAVI on dataset with {N} samples and {K} categories")

    results = compute_multiclass_probit_vi_with_normal_prior(
        labels_train,
        covariates_train,
        labels_test=labels_test,
        covariates_test=covariates_test,
        variational_params_init=None,
        max_n_iterations=max_n_iterations,
    )
    print(f"\n\nPredictive performance over time: \n {results.performance_over_time}")

    meta_data_dict = {
        "user_domain": user_domain,
        "N": N,
        "last_train_idx": n_train_samples,
        "K": K,
        "window_size": window_size,
        "temperature": temperature,
        "max_n_iterations": max_n_iterations,
    }

    ###
    # Save results
    ###

    ensure_dir(save_dir)

    path_to_meta_data = os.path.join(save_dir, f"{user_domain}_meta_data.json")
    write_json(meta_data_dict, path_to_meta_data)

    # UNDO this?
    VI_results = VI_results_from_CAVI_results(results)
    write_VI_results(VI_results, save_dir, user_domain)


if __name__ == "__main__":
    """
    Usage:
        Example:
            python src/categorical_from_binary/experiments/intrusion_detection/main/train_model.py  --user_idx_relative_to_subset 1 --number_of_users_in_subset 10 --start_idx_for_users_in_subset 300 --test_run true
            python src/categorical_from_binary/experiments/intrusion_detection/main/train_model.py  --user_idx_relative_to_subset 1 --number_of_users_in_subset 10 --start_idx_for_users_in_subset 300
    """

    args = _get_argument_parser().parse_args()
    user_idx_relative_to_subset = args.user_idx_relative_to_subset
    number_of_users_in_subset = args.number_of_users_in_subset
    start_idx_for_users_in_subset = args.start_idx_for_users_in_subset
    window_size = args.window_size
    temperature = args.temperature
    max_n_iterations = args.max_n_iterations
    path_to_human_process_start_data = args.path_to_human_process_start_data
    save_dir = args.save_dir
    test_run = args.test_run

    train_model(
        user_idx_relative_to_subset,
        number_of_users_in_subset,
        start_idx_for_users_in_subset,
        window_size,
        temperature,
        max_n_iterations,
        path_to_human_process_start_data,
        save_dir,
        test_run,
    )
