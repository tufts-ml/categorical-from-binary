import collections
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.performance_over_time.results import update_performance_results
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


def construct_performance_over_time_for_MCMC(
    beta_samples_MCMC_without_warmup: NumpyArray3D,
    time_for_MCMC_with_warmup: float,
    covariates_train: NumpyArray2D,
    labels_train: NumpyArray2D,
    covariates_test: NumpyArray2D,
    labels_test: NumpyArray2D,
    link: Link,
    stride: Optional[int] = 1,
    n_warmup_samples: Optional[int] = 0,
    one_beta_sample_has_transposed_orientation: bool = False,
) -> DataFrame:
    """
    Arguments:
        beta_samples_MCMC_without_warmup: array of shape (S,K,M) or (S,M,K), where S is the number of MCMC samples
            (not including warmup), K is the number of categories,
            and M is the number of covariates (including intercept).

            If one_beta_sample_has_transposed_orientation=True, then the shape is taken to be (S,K,M).
            Otherwise, the shape is taken to be (S,M,K).
        time_for_MCMC_with_warmup: time it took to run MCMC.  This includes warmup.
        n_warmup_samples: the number of warmup samples.  ONLY change this from the default (=0)
            if warmup samples were computed separately (then discarded) for some initial percentage
            of `time_for_MCMC_with_warmup`, and then we must post-correct the time estimates.  For instance, we must
            do this when using numpyro's HMC.
        one_beta_sample_has_transposed_orientation : Determines the orientation of beta_samples_MCMC_without_warmup. See beta_samples_MCMC_without_warmup.
    """
    warnings.warn(
        "The beta samples provided should ALREADY have the warmup samples excluded."
    )
    n_useable_samples = np.shape(beta_samples_MCMC_without_warmup)[0]
    pct_time_warming_up = n_warmup_samples / (n_warmup_samples + n_useable_samples)
    estimated_time_for_warmup = pct_time_warming_up * time_for_MCMC_with_warmup
    estimated_time_for_useable_samples = (
        time_for_MCMC_with_warmup - estimated_time_for_warmup
    )

    performance_over_time_as_dict = collections.defaultdict(list)
    for i in range(0, n_useable_samples, int(stride)):
        sample_idx = i + 1
        beta_mean_mcmc_so_far = beta_samples_MCMC_without_warmup[
            :sample_idx, :, :
        ].mean(axis=0)
        if one_beta_sample_has_transposed_orientation:
            beta_mean_mcmc_so_far = beta_mean_mcmc_so_far.T
        seconds_elapsed = (
            estimated_time_for_warmup
            + (sample_idx / n_useable_samples) * estimated_time_for_useable_samples
        )
        update_performance_results(
            performance_over_time_as_dict,
            covariates_train,
            labels_train,
            beta_mean_mcmc_so_far,
            seconds_elapsed,
            link=link,
            covariates_test=covariates_test,
            labels_test=labels_test,
        )
    return pd.DataFrame(performance_over_time_as_dict)
