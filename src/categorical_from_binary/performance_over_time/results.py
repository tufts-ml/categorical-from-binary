import collections
from typing import Optional

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.metrics import compute_metrics
from categorical_from_binary.types import NumpyArray2D


def update_performance_results(
    performance_dict: collections.defaultdict,
    covariates_train: NumpyArray2D,
    labels_train: NumpyArray2D,
    beta_mean: NumpyArray2D,
    secs_elapsed: float,
    link: Link,
    covariates_test: Optional[NumpyArray2D] = None,
    labels_test: Optional[NumpyArray2D] = None,
    update_secs_elapsed: bool = True,
):
    """
    Arguments:
        performance_dict: Must be of the form collections.defaultdict(list)
            This maps strings (metric names) to lists, and each time we update our evaluation,
            we update the dict.
        update_secs_elapsed: bool. Defaults to true, but can be turned off because we may call
            this function multiple times using multiple links (namely we do this for IB-CAVI,
            since we can plug the learned betas into multiple CB models)
    """
    if update_secs_elapsed:
        performance_dict["seconds elapsed"].append(secs_elapsed)
    for mode in ["train", "test"]:
        if mode == "train":
            covariates, labels = covariates_train, labels_train
        elif mode == "test":
            covariates, labels = covariates_test, labels_test
        else:
            raise ValueError("I'm unsure which mode you want to use, train or test")

        cat_probs = construct_category_probs(covariates, beta_mean, link=link)
        metrics = compute_metrics(cat_probs, labels)
        performance_dict[f"{mode} mean likelihood with {link.name}"].append(
            metrics.mean_like
        )
        performance_dict[f"{mode} mean log likelihood with {link.name}"].append(
            metrics.mean_log_like
        )
        performance_dict[f"{mode} accuracy with {link.name}"].append(metrics.accuracy)
    return performance_dict


def get_most_recent_performance_results_as_string(
    performance_dict: collections.defaultdict,
):
    """
    Arguments:
        performance_dict: Must be of the form collections.defaultdict(list)
        This maps strings (metric names) to lists, and each time we update our evaluation,
        we update the dict.
    """
    most_recent_results = ""
    for key, list_of_values in performance_dict.items():
        most_recent_results += f"{key}: {list_of_values[-1]:.03f}\n"
    return most_recent_results
