import collections
from typing import List

import pandas as pd
from pandas.core.frame import DataFrame
from pymc3.variational.approximations import MeanField

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.deprecated.advi_via_bambi.helpers import (
    IntermediateResult,
    get_beta_mean_from_flat_beta_vector,
)
from categorical_from_binary.metrics import compute_metrics
from categorical_from_binary.types import NumpyArray2D


def construct_holdout_performance_over_time_for_legacy_ADVI_via_bambi(
    intermediate_results: List[IntermediateResult],
    model_fitted_by_ADVI: MeanField,
    time_for_ADVI: float,
    labels_train: NumpyArray2D,
    covariates_test: NumpyArray2D,
    labels_test: NumpyArray2D,
    link: Link,
) -> DataFrame:
    """
    Arguments:
        time_for_ADVI: time it took to run ADVI.
        n_advi_samples : number of samples to take when estimating the posterior mean at each landmark iteration
    """
    n_landmark_iterations = len(intermediate_results)
    time_per_landmark_iteration = time_for_ADVI / n_landmark_iterations

    holdout_performance_over_time_as_dict = collections.defaultdict(list)
    for (i, intermediate_result) in enumerate(intermediate_results):
        beta_mean_so_far = get_beta_mean_from_flat_beta_vector(
            intermediate_result.beta_mean_vector,
            model_fitted_by_ADVI,
            labels_train,
        )

        probs = construct_category_probs(
            covariates_test,
            beta_mean_so_far,
            link,
        )
        metrics = compute_metrics(probs, labels_test)
        mean_holdout_log_likelihood = metrics.mean_log_like
        accuracy = metrics.accuracy
        estimated_time = time_per_landmark_iteration * (i + 1)
        holdout_performance_over_time_as_dict["seconds elapsed"].append(estimated_time)
        holdout_performance_over_time_as_dict["mean holdout log likelihood"].append(
            mean_holdout_log_likelihood
        )
        holdout_performance_over_time_as_dict["correct classification rate"].append(
            accuracy
        )
    return pd.DataFrame(holdout_performance_over_time_as_dict)
