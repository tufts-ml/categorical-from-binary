"""
Feature selection 
"""
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


pd.set_option("display.float_format", lambda x: "%.2f" % x)
from pandas.core.frame import DataFrame

from categorical_from_binary.covariate_dict import (
    VariableInfo,
    sorted_covariate_names_from_covariate_dict,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def compute_posterior_prob_of_scaled_neighborhood_around_zero(
    beta_mean: NumpyArray2D,
    beta_stds: Union[NumpyArray1D, NumpyArray2D],
    neighborhood_radius_in_units_of_std_devs: float = 1.0,
    covariate_dict: Optional[Dict[str, VariableInfo]] = None,
    label_dict: Optional[Dict[str, VariableInfo]] = None,
) -> DataFrame:
    """
    If `r` refers to `neighborhood_radius_in_units_of_std_devs`, this computes the  posterior probability of 0 being within
    some epislon-ball around zero,
        P{beta_j in scaled neighborhood around 0) := P_{posterior on beta_j}( - epsilon  <= 0 <= epsilon)
    where epsilon =  r * std(beta_j)

    Note that the variational posterior distribution on betas is normal.

    Reference:
        Li, Q., & Lin, N. (2010). The Bayesian elastic net. Bayesian analysis, 5(1), 151-170,
        Sections 2.5 and 3.

    Arguments:
        beta_mean: has shape (M,K), where M is number of covariates and K is the
            number of categories
        beta_stds: has shape (M,) where M is the number of covariates
            OR has shape (M,K), where M is number of covariates and K is the
            number of categories
    """
    # handle the binary (two-category) case. here it looks like there is only one "category",
    # although that's just a function of how binary probit is structured.
    if beta_mean.ndim == 1:
        beta_mean = beta_mean[:, np.newaxis]

    n_covariates, n_categories = np.shape(beta_mean)

    if beta_stds.ndim == 1:
        beta_stds = np.repeat(beta_stds[:, np.newaxis], n_categories, axis=1)

    posterior_probs_in_neighborhood = np.zeros((n_covariates, n_categories))
    for p in range(n_covariates):
        for k in range(n_categories):
            mean, std = beta_mean[p, k], beta_stds[p, k]
            epsilon = neighborhood_radius_in_units_of_std_devs * std
            posterior_probs_in_neighborhood[p, k] = norm(mean, std).cdf(epsilon) - norm(
                mean, std
            ).cdf(-epsilon)

    if covariate_dict is not None:
        index = sorted_covariate_names_from_covariate_dict(covariate_dict)
    else:
        index = None

    if label_dict is not None:
        columns = list(label_dict.keys())
    else:
        columns = None

    return pd.DataFrame(
        data=posterior_probs_in_neighborhood,
        index=index,
        columns=columns,
    )


def compute_feature_inclusion_data_frame_using_scaled_neighborhood_method(
    beta_mean: NumpyArray2D,
    beta_stds: Union[NumpyArray1D, NumpyArray2D],
    neighborhood_probability_threshold_for_exclusion: float,
    neighborhood_radius_in_units_of_std_devs: float,
    covariate_dict: Optional[Dict[str, VariableInfo]] = None,
    label_dict: Optional[Dict[str, VariableInfo]] = None,
) -> DataFrame:
    """
    If `r` refers to `neighborhood_radius_in_units_of_std_devs`, we first compute the  posterior probability of 0 being within
    some epsilon-ball around zero,
        P({beta_j in scaled neighborhood around 0}) := P_{posterior on beta_j}( - epsilon  <= 0 <= epsilon)
    where epsilon =  r * std(beta_j)

    We then exclude from the model any predictor beta_j such that
        P({beta_j in scaled neighborhood around 0}) >= neighborhood_probability_threshold_for_exclusion.

    Note that the variational posterior distribution on betas is normal.

    Reference:
        Li, Q., & Lin, N. (2010). The Bayesian elastic net. Bayesian analysis, 5(1), 151-170,
        Sections 2.5 and 3.

    Arguments:
        beta_mean: has shape (M,K), where M is number of covariates and K is the
            number of categories
        beta_stds: has shape (M,) where M is the number of covariates
            OR has shape (M,K), where M is number of covariates and K is the
            number of categories
        neighborhood_probability_threshold_for_exclusion: A predictor is excluded if the
            posterior probability of being in a (-std beta_j, +std beta_j)
            exceeds this probability neighborhood_probability_threshold_for_exclusion and is retained otherwise.
    """
    scaled_neighborhood_probs = (
        compute_posterior_prob_of_scaled_neighborhood_around_zero(
            beta_mean,
            beta_stds,
            neighborhood_radius_in_units_of_std_devs,
            covariate_dict,
            label_dict,
        )
    )
    inclusion_matrix_as_boolean = (
        scaled_neighborhood_probs < neighborhood_probability_threshold_for_exclusion
    )
    return (inclusion_matrix_as_boolean).astype(int)
