"""
The purpose of this model is to provide metrics for any categorical model
that produces categorical probabilities, by comparing those probabilities
against the actual label.

Some metrics for performance:
    * mean log likelihood
    * correct classification rate 
    * mean choice rank 
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import scipy

from categorical_from_binary.baserate import compute_probs_for_baserate_model
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.types import NumpyArray2D


@dataclass
class Metrics:
    mean_like: Optional[float] = None
    mean_log_like: Optional[float] = None
    accuracy: Optional[float] = None
    mean_choice_rank: Optional[float] = None


###
# Log Likelihood
###


def _get_probs_of_choices(
    probs: NumpyArray2D,
    labels: NumpyArray2D,
    min_allowable_prob: Optional[float] = None,
):
    """
    Get probabilities of the observed categories (the choices)

    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
        min_allowable_prob: If not None, we effectively take the probabilities over response categories, replace
            each component prob_k with min(min_allowable_prob, prob_k), and then renormalize.
    """
    choices = np.argmax(labels, 1)
    probs_of_choices = np.array(
        [probs[i, choice] for (i, choice) in enumerate(choices)]
    )
    if min_allowable_prob is not None:
        probs_of_choices += min_allowable_prob
    return probs_of_choices


def compute_mean_log_likelihood(
    probs: NumpyArray2D,
    labels: NumpyArray2D,
    min_allowable_prob: Optional[float] = None,
):
    """
    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
        min_allowable_prob: If not None, we effectively take the probabilities over response categories, replace
            each component prob_k with min(min_allowable_prob, prob_k), and then renormalize.
    """
    probs_of_choices = _get_probs_of_choices(probs, labels, min_allowable_prob)
    return np.nanmean(np.log(probs_of_choices))


def compute_mean_likelihood(
    probs: NumpyArray2D,
    labels: NumpyArray2D,
    min_allowable_prob: Optional[float] = None,
):
    """
    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
        min_allowable_prob: If not None, we effectively take the probabilities over response categories, replace
            each component prob_k with min(min_allowable_prob, prob_k), and then renormalize.
    """
    probs_of_choices = _get_probs_of_choices(probs, labels, min_allowable_prob)
    return np.nanmean(probs_of_choices)


def compute_mean_log_likelihood_from_features_and_beta(
    features: NumpyArray2D, beta: NumpyArray2D, labels: NumpyArray2D, link: Link
):
    category_probs = construct_category_probs(features, beta, link)
    return compute_mean_log_likelihood(category_probs, labels)


###
# Correct Classification Rate
###


def compute_accuracy(probs: NumpyArray2D, labels: NumpyArray2D):
    """
    If `S` multiple labels  attained the maximum cat probability,
    the accuracy status for the given observation is 1/S rather than 1. Thus,
    if there are `K` total category probabilities then under a uniform
    category probability distribution, the accuracy will be 1/K.

    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
    """
    # Previous/naive method for computing accuracy below.  We discarded this because np.argmax takes the FIRST index
    # attaining the max.  So for instance under uniform category probabilities it reports
    # the number of times that the chosen response was the 0th category -- obviously a meaningless
    # computation, since indices are assigned to categories arbitrarily.
    #   return np.mean(np.argmax(probs, 1) == np.argmax(labels, 1))

    booleans_giving_attainment_of_max_cat_prob = probs == probs.max(1, keepdims=True)
    num_cats_with_max_cat_prob = np.sum(booleans_giving_attainment_of_max_cat_prob, 1)
    # TODO: Move this conversion WAY higher up in the code, like when we create sparse datasets
    # such as cyber
    if scipy.sparse.issparse(labels):
        labels = labels.astype(bool).todense()

    booleans_stating_whether_choice_attained_max_cat_prob = (
        booleans_giving_attainment_of_max_cat_prob & labels
    )
    # if two labels attained the maximum cat probability, the accuracy status is 1/2 rather than 1.
    accuracy_status_per_observation = np.sum(
        booleans_stating_whether_choice_attained_max_cat_prob
        / num_cats_with_max_cat_prob[:, np.newaxis],
        1,
    )
    return np.mean(accuracy_status_per_observation)


def _compute_balanced_accuracy(probs: NumpyArray2D, labels: NumpyArray2D):
    """
    Experimental function.  Currently not used.

    Reference:
        https://www.michaelchughes.com/papers/HuangEtAl_MLHC_2021.pdf#page=15

    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
    """
    try:
        number_of_examples_selecting_each_category = np.sum(labels, 0)
        booleans_giving_attainment_of_max_cat_prob = probs == probs.max(
            1, keepdims=True
        )
        num_cats_with_max_cat_prob = np.sum(
            booleans_giving_attainment_of_max_cat_prob, 1
        )

        if scipy.sparse.issparse(labels):
            labels = labels.astype(bool).todense()

        booleans_stating_whether_choice_attained_max_cat_prob = (
            booleans_giving_attainment_of_max_cat_prob & labels
        )
        adjusted_true_positive_status = (
            booleans_stating_whether_choice_attained_max_cat_prob
            / num_cats_with_max_cat_prob[:, np.newaxis]
        )
        # `adjusted_true_positive_status` gives 1/S rather than 1 if S different categories tied for the maximum cat prob for that example
        adjusted_accuracy_per_category = (
            np.sum(adjusted_true_positive_status, 0)
            / number_of_examples_selecting_each_category
        )
        # categories with no examples selecting that category are ignored.
        balanced_accuracy = np.nanmean(adjusted_accuracy_per_category)
    except:
        balanced_accuracy = np.nan
    return balanced_accuracy


###
# Choice Ranks
###
#
def compute_choice_ranks(probs: NumpyArray2D, labels: NumpyArray2D):
    """
    Given a set of model probabilities for each category and then actual observed categories ("choices"),
    we report the choice ranks for each sample.

    We compare this to the choice ranks for a baserate model, which simply makes predictions based on the
    empirical frequencies in the training set, completely ignoring the covariates.

    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
    """

    ranks_of_predicted_categories_high_to_low = (-1 * probs).argsort()

    choices = np.argmax(labels, 1)

    n_samples = len(choices)
    choice_ranks_zero_indexed = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        choice_ranks_zero_indexed[i] = np.argwhere(
            ranks_of_predicted_categories_high_to_low[i] == choices[i]
        )[0][0]
    return choice_ranks_zero_indexed + 1


###
# All Metrics
###


def compute_metrics(
    probs: NumpyArray2D,
    labels: NumpyArray2D,
    min_allowable_prob: Optional[float] = None,
) -> Metrics:
    """
    Arguments:
        probs: np.ndarray with shape: (n_samples, n_categories)
            Each row sums to 1
        labels: np.ndarray  with shape: (n_samples, n_categories)
            Each row has exactly one 1.
        min_allowable_prob: If not None, we effectively take the probabilities over response categories, replace
            each component prob_k with min(min_allowable_prob, prob_k), and then renormalize.
    """
    mean_likelihood = compute_mean_likelihood(probs, labels, min_allowable_prob)
    mean_log_likelihood = compute_mean_log_likelihood(probs, labels, min_allowable_prob)
    accuracy = compute_accuracy(probs, labels)
    mean_choice_rank = np.mean(compute_choice_ranks(probs, labels))
    return Metrics(
        mean_likelihood,
        mean_log_likelihood,
        accuracy,
        mean_choice_rank,
    )


###
# Helper functions for conversion to dataframe
###
def append_metrics_dict_for_one_dataset_to_results_dict(
    metrics_dict_for_one_dataset: Dict[str, Metrics],
    results_dict: Optional[Dict] = None,
):
    """
    Arguments:
        metrics_dict_for_one_dataset:  Dict, maps the name of an inference technique to an instance of a Metrics object
        results_dict: Optional[Dict].  Pre-existing results dict.

    Returns:
        results_dict, mapping the Cartesian product (inference_approach x metric name) to a list of values.  If the list contains
        multiple values, then this function must have been called multiple times, successively mutating the results_dict,
        and each item in the list represents the result for a single seed.

    Example:
        metrics_dict_for_one_dataset:
            {'dgp': Metrics(mean_log_like=-0.8681106384500588, accuracy=0.7, mean_choice_rank=1.525),
            'multi_logit_pga_gibbs': Metrics(mean_log_like=-2.020928394329602, accuracy=0.4, mean_choice_rank=2.65),
            'IB_CAVI_plus_BMA': Metrics(mean_log_like=-1.5077224060628591, accuracy=0.45, mean_choice_rank=2.35)}
        results_dict:
            defaultdict(list,
            {'dgp_mean_log_like': [-0.8681106384500588],
             'dgp_accuracy': [0.7],
             'dgp_mean_choice_rank': [1.525],
             'multi_logit_pga_gibbs_mean_log_like': [-2.020928394329602],
             'multi_logit_pga_gibbs_accuracy': [0.4],
             'multi_logit_pga_gibbs_mean_choice_rank': [2.65],
             'IB_CAVI_plus_BMA_mean_log_like': [-1.5077224060628591],
             'IB_CAVI_plus_BMA_accuracy': [0.45],
             'IB_CAVI_plus_BMA_mean_choice_rank': [2.35]})

    """
    if results_dict is None:
        results_dict = defaultdict(list)
    for inference_approach, metrics in metrics_dict_for_one_dataset.items():
        for metric_name, metric_value in asdict(metrics).items():
            results_dict[f"{inference_approach}_{metric_name}"].append(metric_value)
    return results_dict


###
# Report
###
def print_performance_report(
    covariates_test: NumpyArray2D,
    labels_test: NumpyArray2D,
    labels_train: NumpyArray2D,
    beta_star: NumpyArray2D,
    link: Link,
    verbose: bool = True,
) -> None:

    probs_test = construct_category_probs(covariates_test, beta_star, link)
    metrics = compute_metrics(probs_test, labels_test)
    print(f"\n Metrics with {link}: {metrics}")

    probs_test_with_baserate_model = compute_probs_for_baserate_model(
        labels_train, len(labels_test)
    )
    metrics_with_baserate_model = compute_metrics(
        probs_test_with_baserate_model, labels_test
    )
    print(f"\n Metrics with baserate model: {metrics_with_baserate_model}")

    ###
    # Below are some other computations that aren't including in the metrics above.
    ###

    if verbose:
        observed_labels = np.argmax(labels_test, 1)
        predicted_labels = np.argmax(probs_test, 1)

        ### how often is correct decision the dominant one
        most_common_label_test = scipy.stats.mode(observed_labels).mode[0]

        correct_prediction_over_samples = predicted_labels == observed_labels
        percentage_of_time_dominant_category_was_observed = np.mean(
            predicted_labels == most_common_label_test
        )
        print(
            f"The percentage of time the dominant category was observed in test set: {percentage_of_time_dominant_category_was_observed :.03f}"
        )

        percentage_correct_prediction_was_dominant_category = np.mean(
            predicted_labels[correct_prediction_over_samples] == most_common_label_test
        )
        print(
            f"The percentage of time the correct prediction was the dominant category: {percentage_correct_prediction_was_dominant_category:.03f}"
        )

        # Counts of observed vs predicted
        most_common_predicted_labels = Counter(predicted_labels).most_common(15)
        most_common_observed_labels = Counter(observed_labels).most_common(15)

        print(
            f"Most common predicted labels (label_id, count): {most_common_predicted_labels}"
        )
        print(
            f"Most common observed labels (label_id, count): {most_common_observed_labels}"
        )
