import numpy as np
from pandas.core.frame import DataFrame

from categorical_from_binary.types import NumpyArray2D


def exclude_based_on_missingness(
    df: DataFrame, threshold_on_proportion_missing_in_column: float
) -> DataFrame:
    """
    We first remove all columns in which a certain proportion of rows are missing.
    We then remove all rows with any missing data.
    """
    proportion_missing_in_column = df.isnull().mean()
    df_prevalent_columns_only = df.loc[
        :, proportion_missing_in_column < threshold_on_proportion_missing_in_column
    ]
    return df_prevalent_columns_only.dropna(axis="index")


def exclude_based_on_zero_std(df: DataFrame) -> DataFrame:
    return df.drop(df.std()[df.std() == 0.0].index.values, axis=1)


def standardize_numpy_matrix_by_columns(X: NumpyArray2D) -> NumpyArray2D:
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
