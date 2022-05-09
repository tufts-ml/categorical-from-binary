from typing import List, NamedTuple, Tuple

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.evaluate.multiclass import DataType, Measurement, Metric


class HoldoutLogLikes(NamedTuple):
    ib_plus_sdo: List[float]
    ib_plus_do: List[float]


class MisclassificationRates(NamedTuple):
    ib_plus_sdo: List[float]
    ib_plus_do: List[float]


def get_holdout_loglikes(
    measurements_on_all_splits: List[List[Measurement]],
) -> HoldoutLogLikes:
    """
    Arguments:
        measurements_on_all_splits: A list of lists of measurements.
            For the outer list, each element gives results for a different data split.
            For the inner list, each element is a different measurement
    """
    (
        holdout_log_likes_for_ib_plus_sdo,
        holdout_log_likes_for_ib_plus_do,
    ) = get_relevant_metrics_for_cbm_vs_cbc_on_test_sets_for_numerous_data_splits(
        measurements_on_all_splits, relevant_metric=Metric.MEAN_LOG_LIKELIHOOD
    )
    return HoldoutLogLikes(
        holdout_log_likes_for_ib_plus_sdo, holdout_log_likes_for_ib_plus_do
    )


def get_misclassification_rates(
    measurements_on_all_splits: List[List[Measurement]],
) -> MisclassificationRates:
    """
    Arguments:
        measurements_on_all_splits: A list of lists of measurements.
            For the outer list, each element gives results for a different data split.
            For the inner list, each element is a different measurement
    """
    (
        misclassification_rates_for_ib_plus_sdo,
        misclassification_rates_for_ib_plus_do,
    ) = get_relevant_metrics_for_cbm_vs_cbc_on_test_sets_for_numerous_data_splits(
        measurements_on_all_splits, relevant_metric=Metric.MISCLASSIFICATION_RATE
    )
    return MisclassificationRates(
        misclassification_rates_for_ib_plus_sdo, misclassification_rates_for_ib_plus_do
    )


def get_relevant_metrics_for_cbm_vs_cbc_on_test_sets_for_numerous_data_splits(
    measurements_on_all_splits: List[List[Measurement]], relevant_metric: Metric
) -> Tuple[List[float], List[float]]:
    """
    Arguments:
        measurements_on_all_splits: A list of lists of measurements.
            For the outer list, each element gives results for a different data split.
            For the inner list, each element is a different measurement
    """

    def measurement_is_relevant(measurement: Measurement) -> bool:
        return (
            measurement.context.data_type == DataType.TEST
            and measurement.context.metric == relevant_metric
        )

    def measurement_uses_do_formula(measurement: Measurement) -> bool:
        return measurement.context.link_for_category_probabilities == Link.CBC_PROBIT

    def measurement_uses_sdo_formula(measurement: Measurement) -> bool:
        return measurement.context.link_for_category_probabilities == Link.CBM_PROBIT

    relevant_metric_for_ib_plus_sdo = [
        m.value
        for one_split_measurements in measurements_on_all_splits
        for m in one_split_measurements
        if measurement_is_relevant(m) and measurement_uses_sdo_formula(m)
    ]

    relevant_metric_for_ib_plus_do = [
        m.value
        for one_split_measurements in measurements_on_all_splits
        for m in one_split_measurements
        if measurement_is_relevant(m) and measurement_uses_do_formula(m)
    ]
    return relevant_metric_for_ib_plus_sdo, relevant_metric_for_ib_plus_do
