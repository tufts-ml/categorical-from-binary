import bidict
import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import lil_matrix

from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.datasets.cyber.util import (
    compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id,
)
from categorical_from_binary.types import NumpyArray2D
from categorical_from_binary.util import one_hot_encoded_array_from_categorical_indices


def construct_labels(
    df: DataFrame,
    minimal_process_id_by_process_id: bidict,
    window_size: int,
    sparse_representation: bool = True,
) -> NumpyArray2D:
    """
    Arguments:
        df: DataFrame, the return value of `load_human_process_start_df`
        minimal_process_id_by_process_id: the return value of construct_minimal_process_id_by_process_id()
            Among other things, contains information about the number of processes represented
        window_size: Size of lookback window of most recently launched processes that are used in the featurization
    """
    label_ints_original = [int(x[1:]) for x in df["process name"]]
    label_ints_minimal = [
        minimal_process_id_by_process_id[x] for x in label_ints_original
    ]
    number_of_unique_processes = (
        compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
            minimal_process_id_by_process_id
        )
    )
    # To compute `number_of_unique_processes`, we must add one to the maximum value to account for zero-indexing
    labels_minimal_one_hot = one_hot_encoded_array_from_categorical_indices(
        label_ints_minimal,
        number_of_unique_processes,
        sparse_representation,
    )
    return labels_minimal_one_hot[window_size:]


def construct_features(
    df: DataFrame,
    minimal_process_id_by_process_id: bidict,
    window_size: int,
    temperature: float,
    include_intercept: bool,
    sparse_representation: bool = True,
) -> NumpyArray2D:
    """
    Take a lookback window of `window_size` processes, and then featurize as
    exp(-timedelta/temperature) seconds.  This featurization makes sense e.g. because there can be simultaneous
    launches, so it is not possible to impose an order.  It also adds some robustness to minor permutations in
    process launches.

    The window_size could be justified by plotting the distribution on the number of simulatneous
    process launches, and saying that the window size is the whatever-th percentile of that distribution

    Arguments:
        df: DataFrame, the return value of `load_human_process_start_df`
        minimal_process_id_by_process_id: the return value of construct_minimal_process_id_by_process_id()
            Among other things, contains information about the number of processes represented
        window_size: Size of lookback window of most recently launched processes that are used in the featurization
        temperature: Controls how much the strength of influence of previously launched features decays over time.
        include_intercept:
            If True, the 0-th row of the beta matrix will correspond to the intercept, and the 0-th column
            of the features matrix will be a column of all 1's.  If False, neither condition will be true.
    """
    label_ints_original = [int(x[1:]) for x in df["process name"]]
    label_ints_minimal = [
        minimal_process_id_by_process_id[x] for x in label_ints_original
    ]
    times = list(df["time"].values)
    N = len(label_ints_minimal) - window_size
    K = compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
        minimal_process_id_by_process_id
    )
    if sparse_representation:
        features_without_ones_column = lil_matrix((N, K))
    else:
        features_without_ones_column = np.zeros((N, K))
    for i in range(N):
        window_start, curr_idx = i, i + window_size
        reference_time = times[curr_idx]
        lookback_labels = label_ints_minimal[window_start:curr_idx]
        lookback_time_deltas = reference_time - times[window_start:curr_idx]
        lookback_feature_values = np.exp(-lookback_time_deltas / temperature)
        # Note: if the same process was launched multiple times in the lookback window,
        # this code will use the minimum time.
        for label, feature_value in zip(lookback_labels, lookback_feature_values):
            features_without_ones_column[i, label] = feature_value

    if include_intercept:
        features = prepend_features_with_column_of_all_ones_for_intercept(
            features_without_ones_column
        )
    else:
        features = features_without_ones_column
    return features
