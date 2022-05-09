from enum import Enum

import numpy as np

from categorical_from_binary.types import NumpyArray2D


class ShrinkageGroupingStrategy(Enum):
    FREE = 1
    COVARIATES = 2
    FREE_INTERCEPTS_BUT_GROUPS_FOR_OTHER_COVARIATES = 2


def make_shrinkage_groups(
    shrinkage_grouping_strategy: ShrinkageGroupingStrategy, beta_mean: NumpyArray2D
) -> NumpyArray2D:
    num_covariates, num_categories = np.shape(beta_mean)
    if shrinkage_grouping_strategy == ShrinkageGroupingStrategy.FREE:
        indices = [i for i in range(num_covariates * num_categories)]
    elif (
        shrinkage_grouping_strategy
        == ShrinkageGroupingStrategy.FREE_INTERCEPTS_BUT_GROUPS_FOR_OTHER_COVARIATES
    ):
        indices_for_intercepts = [i for i in range(num_categories)]
        indices_for_other_covariates = [
            i + num_categories
            for i in range((num_covariates - 1))
            for j in range(num_categories)
        ]
        indices = indices_for_intercepts + indices_for_other_covariates
    elif shrinkage_grouping_strategy == ShrinkageGroupingStrategy.COVARIATES:
        indices = [i for i in range(num_covariates) for j in range(num_categories)]
    else:
        raise ValueError("I don't understand the shrinkage grouping strategy")
    return np.array(indices).reshape(num_covariates, num_categories)
