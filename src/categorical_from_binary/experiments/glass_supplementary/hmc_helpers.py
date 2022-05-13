"""
Just a wrapper around some of the functionality provided by categorical_from_binary.hmc
that helps to analyze the glass identification data.  

A natural question: Can't we just absorb all this into the hmc subpackage directly?
Looking at this a while after this was written, it seems that the core purpose for 
this being located here rather than in the hmc subpackage may be that we generally prefer
to represent HMC beta samples as an array of shape (S,M,K), where S is the number of Monte Carlo samples,
M is the number of covariates, and K is the number of categories, but the hmc subpackage orients
this same object as (S,K,M). 
"""
from typing import Union

import jaxlib
import numpy as np

from categorical_from_binary.hmc.consts import (
    CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE,
)
from categorical_from_binary.hmc.core import (
    CategoricalModelType,
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


def get_beta_samples_for_categorical_model_via_HMC(
    covariates: NumpyArray2D,
    labels: NumpyArray2D,
    categorical_model_type: CategoricalModelType,
    num_warmup_samples: int,
    num_mcmc_samples: int,
    prior_mean: float = 0.0,
    prior_stddev: float = 1.0,
    random_seed: int = 1,
) -> NumpyArray3D:
    """
    Arguments:
        covariates: np.array of shape (N,M), where N is the number of samples and M
            is the number of covariates
        labels: np.array of shape (N,K), where N is the number of samples and K
            is the number of categories

    Returns:
        beta samples of shape (S,M,K), where S is the number of Monte Carlo samples,
        M is the number of covariates, and K is the number of categories

    """
    n_samples = np.shape(labels)[0]
    Nseen_list = [n_samples]
    betas_SLM_by_N = run_nuts_on_categorical_data(
        num_warmup_samples,
        num_mcmc_samples,
        Nseen_list,
        create_categorical_model,
        categorical_model_type,
        labels,
        covariates,
        prior_mean,
        prior_stddev,
        random_seed,
    )
    betas_SKM = betas_SLM_by_N[n_samples]
    return np.swapaxes(betas_SKM, 1, 2)


def get_mean_log_like_from_beta_samples(
    beta_samples: Union[NumpyArray3D, jaxlib.xla_extension.DeviceArray],
    covariates: NumpyArray2D,
    labels: NumpyArray2D,
    categorical_model_type: CategoricalModelType,
) -> float:
    """
    Arguments:
        beta samples: has shape (S,M,K), where S is the number of Monte Carlo samples,
            M is the number of covariates, and K is the number of categories
        covariates: np.array of shape (N,M), where N is the number of samples and M
            is the number of covariates
        labels: np.array of shape (N,K), where N is the number of samples and K
            is the number of categories
    """
    # jax's DeviceArray's don't support item assignment, so if that's the type of the beta samples
    # convert it to numpy, so that the functions for computing category probabilities in our
    # data generation module can run without raising an error.
    if isinstance(beta_samples, jaxlib.xla_extension.DeviceArray):
        beta_samples = beta_samples.to_py()
    beta_mean = beta_samples.mean(axis=0)
    category_probability_function = CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE[
        categorical_model_type
    ]
    category_probs = category_probability_function(covariates, beta_mean)
    choice_probs = category_probs[np.where(labels)]
    return np.mean(np.log(choice_probs))


def get_accuracy_from_beta_samples(
    beta_samples: Union[NumpyArray3D, jaxlib.xla_extension.DeviceArray],
    covariates: NumpyArray2D,
    labels: NumpyArray2D,
    categorical_model_type: CategoricalModelType,
) -> float:
    """
    Arguments:
        beta samples: has shape (S,M,K), where S is the number of Monte Carlo samples,
            M is the number of covariates, and K is the number of categories
        covariates: np.array of shape (N,M), where N is the number of samples and M
            is the number of covariates
        labels: np.array of shape (N,K), where N is the number of samples and K
            is the number of categories
    """
    # jax's DeviceArray's don't support item assignment, so if that's the type of the beta samples
    # convert it to numpy, so that the functions for computing category probabilities in our
    # data generation module can run without raising an error.
    if isinstance(beta_samples, jaxlib.xla_extension.DeviceArray):
        beta_samples = beta_samples.to_py()
    beta_mean = beta_samples.mean(axis=0)
    category_probability_function = CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE[
        categorical_model_type
    ]
    category_probs = category_probability_function(covariates, beta_mean)
    return np.mean(np.argmax(category_probs, 1) == np.where(labels)[1])


# TODO: Align this with `get_holdout_loglikes` from categorical_from_binary.experiments.glass_supplementary.analysis
