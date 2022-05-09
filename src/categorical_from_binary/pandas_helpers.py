from typing import List, Union

from pandas.core.frame import DataFrame


def filter_df_rows_by_column_values(
    df: DataFrame, col: str, values: Union[str, List[str]]
) -> DataFrame:
    """
    Filter out rows of a pandas dataframe based on the presence of value(s) in a column
    """
    if isinstance(values, str):
        values = [values]
    return df[~df[col].isin(values)]


def filter_df_rows_by_column_value_starts(
    df: DataFrame, col: str, start: str
) -> DataFrame:
    """
    Filter out rows of a pandas dataframe based on the presence of value(s) in a column
    which begin with something
    """
    return df[df[col].str.startswith(start)]


def keep_df_rows_by_column_values(
    df: DataFrame, col: str, values: Union[str, List[str]]
) -> DataFrame:
    """
    Keep rows of a pandas dataframe based on the presence of value(s) in a column
    """
    if isinstance(values, str):
        values = [values]
    return df[df[col].isin(values)]
