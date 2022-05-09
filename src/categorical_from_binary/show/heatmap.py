import textwrap
from typing import Callable, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


matplotlib.rc_file_defaults()

from pandas.core.frame import DataFrame

from categorical_from_binary.types import NumpyArray2D


def plot_heatmap(
    beta_mean: NumpyArray2D,
    feature_inclusion_df: DataFrame,
    white_out_excluded_covariates: bool,
    x_ticks_fontsize: float = 8,
    function_to_clean_covariate_names: Optional[Callable] = None,
    give_each_intercept_term_its_own_heatmap_color: bool = False,
) -> None:
    """
    Plot heatmap of model's regression weights across covariates and categories.
    Can zero out all non-included features.

    Arguments:
        beta_mean: has shape (M,K), where M is number of covariates and K is the
            number of categories
        feature_inclusion_df: has shape (M,K), where M is number of covariates and K is the
            number of categories
    """
    # handle the binary (two-category) case. here it looks like there is only one "category",
    # although that's just a function of how binary probit is structured.
    if beta_mean.ndim == 1:
        beta_mean = beta_mean[:, np.newaxis]

    if feature_inclusion_df is not None:
        matrix_to_show_transposed = feature_inclusion_df * beta_mean
    else:
        matrix_to_show_transposed = beta_mean

    # tranpose this so we show the labels on the y-axis
    matrix_to_show = matrix_to_show_transposed.T

    if white_out_excluded_covariates:
        matrix_to_show[matrix_to_show == 0.0] = np.nan

    ### Get covariate names (shorten them so plot doesn't eat them)
    covariate_names = list(feature_inclusion_df.index)
    if function_to_clean_covariate_names is not None:
        covariate_names = [
            function_to_clean_covariate_names(x) for x in covariate_names
        ]
    label_names = list(feature_inclusion_df.columns)

    cmap = matplotlib.cm.viridis

    # get boundaries for heatmap
    n_bins = 10
    spacing = int(100 / n_bins)
    values = np.ndarray.flatten(beta_mean).tolist()
    percentiles = np.percentile(
        values, q=list(range(0, 100 + spacing, spacing))
    ).tolist()
    if give_each_intercept_term_its_own_heatmap_color:
        intercepts = beta_mean[0, :].tolist()
        percentiles = list(set(percentiles + intercepts))
    bounds = np.round(np.sort(percentiles + [0]), 3)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ### Make image plot

    fig, ax = plt.subplots()
    im = ax.imshow(matrix_to_show, cmap=cmap, norm=norm)
    plt.xticks(
        np.arange(len(covariate_names)),
        covariate_names,
        rotation=80,
        fontsize=x_ticks_fontsize,
    )
    plt.yticks(np.arange(len(label_names)), label_names)

    # try wrapping long names
    f = lambda x: textwrap.fill(x.get_text(), 30)
    ax.set_xticklabels(map(f, ax.get_xticklabels()))

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
