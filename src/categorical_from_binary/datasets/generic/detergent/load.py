import random

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.util import (
    construct_standardized_design_matrix,
    one_hot_encoded_array_from_categorical_indices,
)


###
# LOAD DETERGENT DATA
###


def load_detergent_data() -> DataFrame:
    """
    A categorial regression dataset.  Referenced in an R package for multionomial probit
    regression.

    See: https://github.com/kosukeimai/MNP/tree/master/data
    """
    PATH_TO_DETERGENT_DATA = "./data/real_data/detergent/detergent.txt"
    return pd.read_csv(PATH_TO_DETERGENT_DATA, sep=" ")


###
# CONSTRUCT DATA SPLITS
###

# TODO: The fucntion below is identical for both glass ID and detergent
def construct_detergent_data_split(
    detergent_data: DataFrame,
    pct_training: int,
    standardize_design_matrix: bool,
    random_seed: int,
) -> SplitDataset:
    """
    Arguments:
        detergent_data: DataFrame.  Return value of load_detergent_data()

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
    ) = construct_detergent_data_split_and_return_individual_components(
        detergent_data,
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


def construct_detergent_data_split_and_return_individual_components(
    detergent_data: DataFrame,
    pct_training: int,
    standardize_design_matrix: bool,
    random_seed: int,
):
    """
    Arguments:
        detergent data: np.array. First 9 columns are covariates.
            10th column is the labels.  For structure, see
            https://github.com/p-sama/Glass-Classification/blob/master/glass.csv

    Returns:
        Matches the return value of `generate_categorical_data_with_covariates`
        in categorical_from_binary.categorial_models.hmc.generate
    """
    data = detergent_data
    N = np.shape(data)[0]

    possible_choices = list(set(data["choice"]))

    labels = np.zeros(N).astype(int)
    for i in range(N):
        curr_choice = data["choice"][i]
        labels[i] = possible_choices.index(curr_choice)

    covariates_no_intercept = data.iloc[:, 1:].to_numpy()
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

    K = len(possible_choices)
    y_train__one_hot_NK = one_hot_encoded_array_from_categorical_indices(
        labels[train_idxs], K
    )
    y_test__one_hot_NK = one_hot_encoded_array_from_categorical_indices(
        labels[test_idxs], K
    )
    x_train_NM = design_matrix[train_idxs, :]
    x_test_NM = design_matrix[test_idxs, :]

    return y_train__one_hot_NK, y_test__one_hot_NK, x_train_NM, x_test_NM
