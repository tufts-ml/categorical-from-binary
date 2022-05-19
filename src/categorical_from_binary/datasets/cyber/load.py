from typing import Tuple

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
from categorical_from_binary.pandas_helpers import keep_df_rows_by_column_values
from categorical_from_binary.types import NumpyArray2D


def construct_process_start_features_and_labels_for_one_cyber_user(
    path_to_human_process_start_data: str,
    subset_initial_user_idx_when_sorting_most_to_fewest_events: int,
    subset_number_of_users: int,
    user_idx_relative_to_subset: int,
    window_size: int,
    temperature: float,
    include_intercept: bool,
) -> Tuple[NumpyArray2D, NumpyArray2D, str]:
    """
    The number of processes/categories represented (i.e. the number of columns of the one-hot-encoded `labels`)
    will be given by the number of unique categories launched across a subset of users, which is determined
    by grabbing data from users indexed from ` subset_initial_user_idx_when_sorting_most_to_fewest_events`
    to that value plus `subset_number_of_users`.
    """
    df_all = load_human_process_start_df(path_to_human_process_start_data)

    # get a subset of all user domains, from which we determine the number of categories to use for processing.
    sorted_sample_sizes_for_users = compute_sorted_sample_sizes_for_users(df_all)
    user_domains_in_subset = sorted_sample_sizes_for_users.index[
        subset_initial_user_idx_when_sorting_most_to_fewest_events : subset_initial_user_idx_when_sorting_most_to_fewest_events
        + subset_number_of_users
    ]
    df_subset = keep_df_rows_by_column_values(
        df_all, col="user@domain", values=user_domains_in_subset
    )
    minimal_process_id_by_process_id = construct_minimal_process_id_by_process_id(
        df_subset
    )
    # `K` is the number of unique processes
    K = compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
        minimal_process_id_by_process_id
    )
    print(
        f"The total number of distinct processes started in this subset of users is {K}."
    )

    ### Get data for one user
    sorted_sample_sizes_for_users = compute_sorted_sample_sizes_for_users(df_all)
    user_domain = user_domains_in_subset[user_idx_relative_to_subset]
    print(f"The user domain whose data is being grabbed is {user_domain}.")
    df = keep_df_rows_by_column_values(df_all, col="user@domain", values=user_domain)
    labels = construct_labels(df, minimal_process_id_by_process_id, window_size)
    features = construct_features(
        df,
        minimal_process_id_by_process_id,
        window_size,
        temperature,
        include_intercept=include_intercept,
    )
    return features, labels, user_domain
