"""
Load in glass identification dataset

Output is currently (8/2/2021) aligned with categorical_from_binary.hmc.generate,
as both are processed by probit HMC demos and experiments.
"""

import random

import numpy as np

from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.util import (
    construct_standardized_design_matrix,
    one_hot_encoded_array_from_categorical_indices,
)


def construct_glass_identification_data_split(
    glass_identification_data: np.array,
    pct_training: int,
    standardize_design_matrix: bool,
    random_seed: int,
) -> SplitDataset:
    """
    Arguments:
        glass_identification_data: np.array. First 9 columns are covariates.
            10th column is the labels.  For structure, see
            https://github.com/p-sama/Glass-Classification/blob/master/glass.csv

    Returns:
        SplitDataset.  Matches the return value of generate_multiclass_regression_dataset
        from  categorical_from_binary.data_generation.bayes_multiclass_reg which is used
        in demos of VI's quality.
    """
    (
        y_train__one_hot_NK,
        y_test__one_hot_NK,
        x_train_NM,
        x_test_NM,
    ) = construct_glass_identification_data_split_and_return_individual_components(
        glass_identification_data,
        pct_training,
        standardize_design_matrix,
        random_seed,
    )

    # Prep training data
    covariates_train = x_train_NM
    labels_train = y_train__one_hot_NK

    covariates_test = x_test_NM
    labels_test = y_test__one_hot_NK

    return SplitDataset(covariates_train, labels_train, covariates_test, labels_test)


def construct_glass_identification_data_split_and_return_individual_components(
    glass_identification_data: np.array,
    pct_training: int,
    standardize_design_matrix: bool,
    random_seed: int,
):
    """
    Arguments:
        glass_identification_data: np.array. First 9 columns are covariates.
            10th column is the labels.  For structure, see
            https://github.com/p-sama/Glass-Classification/blob/master/glass.csv

    Returns:
        Matches the return value of `generate_categorical_data_with_covariates`
        in categorical_from_binary.categorial_models.hmc.generate
    """
    data = glass_identification_data
    N = np.shape(data)[0]

    ORIGINAL_LABELS_BY_NEW_LABELS = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7}
    # we relabel the categories to (a) be zero-indexed and (b) have no unused categories
    original_labels = data[:, -1].astype(int)
    labels = np.zeros(N).astype(int)
    for new_label, original_label in ORIGINAL_LABELS_BY_NEW_LABELS.items():
        labels[np.where(original_labels == original_label)] = new_label

    covariates_no_intercept = data[:, :-1]
    if standardize_design_matrix:
        covariates_no_intercept = construct_standardized_design_matrix(
            covariates_no_intercept
        )
    design_matrix = np.insert(covariates_no_intercept, 0, 1, axis=1)

    # train test split
    N_train = int(N * pct_training)
    random.seed(random_seed)
    train_idxs = random.sample(range(N), N_train)
    test_idxs = list(set(range(N)).difference(set(train_idxs)))

    K = len(ORIGINAL_LABELS_BY_NEW_LABELS)
    y_train__one_hot_NK = one_hot_encoded_array_from_categorical_indices(
        labels[train_idxs], K
    )
    y_test__one_hot_NK = one_hot_encoded_array_from_categorical_indices(
        labels[test_idxs], K
    )
    x_train_NM = design_matrix[train_idxs, :]
    x_test_NM = design_matrix[test_idxs, :]

    return y_train__one_hot_NK, y_test__one_hot_NK, x_train_NM, x_test_NM


def load_glass_identification_data():
    PATH_TO_DATA = "data/real_data/glass_identification/glass.csv"
    return np.genfromtxt(PATH_TO_DATA, delimiter=",", skip_header=1)
