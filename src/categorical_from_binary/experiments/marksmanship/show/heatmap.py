import re

import matplotlib


matplotlib.rc_file_defaults()

from pandas.core.frame import DataFrame

from categorical_from_binary.show.heatmap import plot_heatmap
from categorical_from_binary.types import NumpyArray2D


def _clean_entry_if_its_a_string_of_the_phrase_n_trials_back(init_entry: str) -> str:
    strip_string = " trials back"
    if isinstance(init_entry, str) and init_entry.endswith(strip_string):
        desired_string_plus_digit = re.sub(strip_string, "", init_entry)
        return desired_string_plus_digit[:-2]
    else:
        return init_entry


def plot_heatmap_for_marksmanship(
    beta_mean: NumpyArray2D,
    feature_inclusion_df: DataFrame,
    white_out_excluded_covariates: bool,
    x_ticks_fontsize: float = 10,
    give_each_intercept_term_its_own_heatmap_color: bool = False,
) -> None:
    """
    Currently, we zero-out all non-included features.

    Arguments:
        beta_mean: has shape (M,K), where M is number of covariates and K is the
            number of categories
        feature_inclusion_df: has shape (M,K), where M is number of covariates and K is the
            number of categories
    """
    return plot_heatmap(
        beta_mean,
        feature_inclusion_df,
        white_out_excluded_covariates,
        x_ticks_fontsize,
        function_to_clean_covariate_names=_clean_entry_if_its_a_string_of_the_phrase_n_trials_back,
        give_each_intercept_term_its_own_heatmap_color=give_each_intercept_term_its_own_heatmap_color,
    )
