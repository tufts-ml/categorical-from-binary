from enum import Enum
from typing import Dict, List, NamedTuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def get_sklearn_category_probabilities(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    **kwargs,
):
    """
    We use these to compare quality of models.
    sklearn doesn't hardcode the beta for the last category to be 0
    (not sure how it gets away with this.  mean overparametrization?)

    However, we CAN compare the category probabilities instead.
    """
    # sklearn doesn't want the all-ones column
    features_without_all_ones_column = features[:, 1:]
    choices = np.nonzero(labels)[1]  # labels (one-hot-encoded) as class indicators
    lr = LogisticRegression(random_state=0, **kwargs).fit(
        features_without_all_ones_column,
        choices,
    )
    # sklearn_betas = np.transpose(np.hstack((lr.intercept_[:, np.newaxis], lr.coef_)))
    # sklearn_betas has shape (n_features+1, n_categories)
    # ugh, hard to compare because sklearn doesn't hardcode the beta for hte last category to be 0.  wtf. how does it do this?
    # we could perhaps compare the predicted probabilities
    sklearn_category_probs = lr.predict_proba(features_without_all_ones_column)
    return sklearn_category_probs


def compute_choice_probs_for_multiclass_regression(
    features: NumpyArray2D,
    labels: NumpyArray2D,
    beta_expected: NumpyArray2D,
    link_for_category_probabilities: Link,
) -> float:
    # `choices` converts one-hot encoded representation to index of selected category
    choices = np.argmax(labels, 1)
    category_probs = construct_category_probs(
        features, beta_expected, link_for_category_probabilities
    )
    return [category_probs[i, choice] for (i, choice) in enumerate(choices)]


class Metric(Enum):
    MEAN_LOG_LIKELIHOOD = 1
    MEAN_LIKELIHOOD = 2
    MISCLASSIFICATION_RATE = 3


EVALUATION_FUNCTIONAL_BY_METRIC_FOR_SOME_METRICS = {
    Metric.MEAN_LIKELIHOOD: lambda choice_probs: np.nanmean(choice_probs),
    Metric.MEAN_LOG_LIKELIHOOD: lambda choice_probs: np.nanmean(np.log(choice_probs)),
}


def compute_mean_likelihood_for_multiclass_regression(
    features: NumpyArray2D,
    labels: NumpyArray2D,
    beta_expected: NumpyArray2D,
    link_for_category_probabilities: Link,
    focal_choice: Optional[int] = None,
) -> float:
    return compute_metric_for_multiclass_regression(
        features,
        labels,
        beta_expected,
        link_for_category_probabilities,
        Metric.MEAN_LIKELIHOOD,
        focal_choice,
    )


def compute_mean_log_likelihood_for_multiclass_regression(
    features: NumpyArray2D,
    labels: NumpyArray2D,
    beta_expected: NumpyArray2D,
    link_for_category_probabilities: Link,
    focal_choice: Optional[int] = None,
) -> float:
    return compute_metric_for_multiclass_regression(
        features,
        labels,
        beta_expected,
        link_for_category_probabilities,
        Metric.MEAN_LOG_LIKELIHOOD,
        focal_choice,
    )


def compute_metric_for_multiclass_regression(
    features: NumpyArray2D,
    labels: NumpyArray2D,
    beta_expected: NumpyArray2D,
    link_for_category_probabilities: Link,
    metric: Metric,
    focal_choice: Optional[int] = None,
) -> float:
    """
    Arguments:
        focal_choice:
            Optional.  If present, only return the mean choice probs for only when the label was
            the `focal_choice`.  Only relevant for Metric. MEAN_LOG_LIKELIHOOD and Metric.MEAN_LIKELIHOOD
    """
    # `choices` converts one-hot encoded representation to index of selected category
    if metric == Metric.MEAN_LIKELIHOOD or metric == Metric.MEAN_LOG_LIKELIHOOD:
        choices = np.argmax(labels, 1)
        choice_probs = compute_choice_probs_for_multiclass_regression(
            features, labels, beta_expected, link_for_category_probabilities
        )
        if focal_choice is not None:
            choice_probs_to_use = list(np.array(choice_probs)[choices == focal_choice])
        else:
            choice_probs_to_use = choice_probs

        functional = EVALUATION_FUNCTIONAL_BY_METRIC_FOR_SOME_METRICS[metric]
        return functional(choice_probs_to_use)
    elif metric == Metric.MISCLASSIFICATION_RATE:
        category_probs = construct_category_probs(
            features, beta_expected, link_for_category_probabilities
        )
        return np.mean(np.argmax(category_probs, 1) != np.where(labels)[1])
    else:
        raise ValueError(f"Not sure how to handle metric {metric}")


class BetaType(Enum):
    VARIATIONAL_POSTERIOR_MEAN = 1
    GROUND_TRUTH = 2


def evaluate_multiclass_regression_with_beta_estimate(
    features: NumpyArray2D,
    labels: NumpyArray2D,
    beta_estimate: NumpyArray2D,
    link_for_category_probabilities: Link,
    metric: Metric,
    verbose: bool = True,
    beta_type: BetaType = BetaType.VARIATIONAL_POSTERIOR_MEAN,
) -> float:
    """
    Arguments:
        link_for_category_probabilities: Link function of the ***variational model***; the link function of the data
            generating process could be different
    """
    # TODO: Improve this to use the posterior predictive.
    metric_variational_model = compute_metric_for_multiclass_regression(
        features,
        labels,
        beta_estimate,
        link_for_category_probabilities,
        metric,
    )

    if verbose:
        print(
            f"The {metric} from plugging the {beta_type} beta  "
            f"into the category probabilities of {link_for_category_probabilities} is {metric_variational_model:.03}"
        )
    return metric_variational_model


def evaluate_sklearn_on_multiclass_regression(
    features: NumpyArray2D,
    labels: NumpyArray2D,
    metric: Metric,
    verbose: bool = True,
    **kwargs_for_sklearn,
) -> float:

    if (
        not metric == Metric.MEAN_LIKELIHOOD
        and not metric == Metric.MEAN_LOG_LIKELIHOOD
    ):
        raise NotImplementedError

    # TODO: This block of code violates DRY; this logic is already written up elsewhere; call it here.
    choices = np.argmax(
        labels, 1
    )  # converts one-hot encoded representation to index of selected category

    category_probs_sklearn = get_sklearn_category_probabilities(
        features, labels, **kwargs_for_sklearn
    )
    category_probs_of_choices_sklearn = [
        category_probs_sklearn[i, choice] for (i, choice) in enumerate(choices)
    ]

    functional = EVALUATION_FUNCTIONAL_BY_METRIC_FOR_SOME_METRICS[metric]
    metric_sklearn = functional(category_probs_of_choices_sklearn)

    if verbose:
        print(f"The {metric} for the sklearn model is {metric_sklearn:.03}")
    return metric_sklearn


class DataType(Enum):
    TRAIN = 1
    TEST = 2


class MeasurementContext(NamedTuple):
    metric: Metric
    beta_type: BetaType
    link_for_category_probabilities: Link
    link_for_generating_data: Link
    data_type: DataType
    other_info: str


class Measurement(NamedTuple):
    value: float
    context: MeasurementContext


def take_measurements_comparing_CBM_and_CBC_estimators(
    split_dataset: SplitDataset,
    beta_mean: NumpyArray2D,
    link_for_generating_data: Optional[Link] = None,
    beta_ground_truth: Optional[NumpyArray2D] = None,
    compare_to_sklearn: bool = False,
    verbose: bool = True,
    **kwargs_for_sklearn,
) -> List[Measurement]:
    """
    The CBM estimator is the Categorical From Binary via Marginalization estimator (see arxiv paper)
    The CBC estimator is the Categorical From Binary via Conditioning estimator (see arxiv paper)
    """

    dataset_parts = {
        DataType.TRAIN: (split_dataset.covariates_train, split_dataset.labels_train),
        DataType.TEST: (split_dataset.covariates_test, split_dataset.labels_test),
    }

    measurements = []
    for data_type, (covariates, labels) in dataset_parts.items():
        print(f"\n\n---Now running evaluations on {data_type} data---")
        for metric in Metric:
            print(f"")
            # first compute ground truth, if one is provided
            if beta_ground_truth is not None:
                beta_type = BetaType.GROUND_TRUTH
                link_for_category_probabilities = link_for_generating_data
                metric_value = evaluate_multiclass_regression_with_beta_estimate(
                    covariates,
                    labels,
                    beta_ground_truth,
                    link_for_category_probabilities,
                    metric=metric,
                    beta_type=beta_type,
                )

                measurements.append(
                    Measurement(
                        metric_value,
                        MeasurementContext(
                            metric,
                            beta_type,
                            link_for_category_probabilities,
                            link_for_generating_data,
                            data_type,
                            other_info="",
                        ),
                    )
                )

            # now compute metrics with the variational approximations, using different constructions for the
            # category probabilities
            beta_type = BetaType.VARIATIONAL_POSTERIOR_MEAN
            link_for_category_probabilities = Link.CBC_PROBIT

            metric_value = evaluate_multiclass_regression_with_beta_estimate(
                covariates,
                labels,
                beta_mean,
                link_for_category_probabilities,
                metric=metric,
                beta_type=beta_type,
                verbose=verbose,
            )
            measurements.append(
                Measurement(
                    metric_value,
                    MeasurementContext(
                        metric,
                        beta_type,
                        link_for_category_probabilities,
                        link_for_generating_data,
                        data_type,
                        other_info="",
                    ),
                )
            )

            # TODO: need to compute the actual posterior predictive for this;
            # i.e. integrate over our uncertainty about model parameters.
            link_for_category_probabilities = Link.CBM_PROBIT
            metric_value = evaluate_multiclass_regression_with_beta_estimate(
                covariates,
                labels,
                beta_mean,
                link_for_category_probabilities,
                metric=metric,
                beta_type=beta_type,
                verbose=verbose,
            )
            measurements.append(
                Measurement(
                    metric_value,
                    MeasurementContext(
                        metric,
                        beta_type,
                        link_for_category_probabilities,
                        link_for_generating_data,
                        data_type,
                        other_info="",
                    ),
                )
            )

            if compare_to_sklearn:
                # TODO: we don't currently create `measurements` for sklearn; just printouts.
                evaluate_sklearn_on_multiclass_regression(
                    covariates, labels, metric, verbose, **kwargs_for_sklearn
                )

    return measurements


def form_dict_mapping_measurement_context_to_values(
    measurements: List[Measurement],
) -> Dict[MeasurementContext, List[float]]:
    """
    given a set of measurements, find the set of unique measurement contexts,
    and collect all the values for that measurement context into a list
    """
    contexts = [m.context for m in measurements]
    context_space = list(set(contexts))

    measurement_context_to_values = {}

    for context in context_space:
        measurements_filtered = list(
            filter(lambda x: x.context == context, measurements)
        )
        values = [m.value for m in measurements_filtered]
        measurement_context_to_values[context] = values

    return measurement_context_to_values
