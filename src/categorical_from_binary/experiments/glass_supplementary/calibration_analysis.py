from typing import Dict, List, NamedTuple

import seaborn as sns


sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.results import ResultsOnOneSplit


class CalibrationCurveResults(NamedTuple):
    """
    For each bucket with some mean predicted probability, gives the fraction of actual positives
    observed.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve
    """

    fraction_of_positives: float
    mean_predicted_prob: float


LINKS_FOR_CATEGORY_PROBABILITIES = [Link.CBC_PROBIT, Link.CBM_PROBIT]


def get_calibration_curve_results_on_one_data_split(
    results_on_one_split: ResultsOnOneSplit, n_bins: int
) -> Dict[Link, CalibrationCurveResults]:
    """
    We use the "quantile" strategy rather than the default strategy to ensure that
        a) all bins are populated (otherwise the number of points on the curve could differ
            for CBC and CBM, which can complicate  downstream metrics, like sum of squared errors)
        b) get more towards a notion of expected calibration error, but locating points in places
            on the x-axis (predicted probabilities) that the model actually PUTS points!
    """
    covariates, labels = (
        results_on_one_split.split_dataset.covariates_test,
        results_on_one_split.split_dataset.labels_test,
    )
    choices = np.argmax(labels, 1)

    calibration_curve_results_by_link_for_category_probabilities = {}

    for link_for_category_probabilities in LINKS_FOR_CATEGORY_PROBABILITIES:
        beta = results_on_one_split.beta_mean

        category_probs = construct_category_probs(
            covariates,
            beta,
            link_for_category_probabilities,
        )

        choice_probs = np.array(
            [category_probs[i, choice] for (i, choice) in enumerate(choices)]
        )
        non_choice_probs = np.delete(
            category_probs, [(i, choice) for (i, choice) in enumerate(choices)]
        )

        choice_indicators = np.array(
            [1] * len(choice_probs) + [0] * len(non_choice_probs)
        )
        probs = np.hstack((choice_probs, non_choice_probs))

        fraction_of_positives, mean_predicted_prob = calibration_curve(
            choice_indicators, probs, n_bins=n_bins, strategy="quantile"
        )
        calibration_curve_results = CalibrationCurveResults(
            fraction_of_positives, mean_predicted_prob
        )
        calibration_curve_results_by_link_for_category_probabilities[
            link_for_category_probabilities
        ] = calibration_curve_results

    return calibration_curve_results_by_link_for_category_probabilities


def compute_sum_of_squared_calibration_error_by_link_on_one_data_split(
    calibration_curve_results_by_link_for_category_probabilities: Dict[
        Link, CalibrationCurveResults
    ]
) -> Dict[Link, float]:
    """
    For each point sampled on the calibration curve, we compute the squared vertical distance to the y=x line.
    We sum these up across sample points to get an overall metric of calibration error.
    """
    sse_by_link = {}
    for link in LINKS_FOR_CATEGORY_PROBABILITIES:
        ccr = calibration_curve_results_by_link_for_category_probabilities[link]
        x = ccr.mean_predicted_prob
        y = ccr.fraction_of_positives
        sse = np.sum(
            (x - y) ** 2
        )  # sum of squared vertical distance from the y=x line of perfect calibration
        sse_by_link[link] = sse
    return sse_by_link


def compute_sum_of_squared_calibration_errors_by_link_from_results_on_many_splits(
    results_on_many_splits: List[ResultsOnOneSplit],
    n_bins: int,
) -> Dict[Link, List[float]]:
    """
    Returns:
        a dictionary mapping the category probability link to a list of sum of squared calibration errors.
        Each element in the list is that error for one data split
    """
    sses_by_link = {link: [] for link in LINKS_FOR_CATEGORY_PROBABILITIES}
    for results_on_one_split in results_on_many_splits:
        calibration_curve_results_by_link_for_category_probabilities = (
            get_calibration_curve_results_on_one_data_split(
                results_on_one_split, n_bins
            )
        )
        sse_by_link = (
            compute_sum_of_squared_calibration_error_by_link_on_one_data_split(
                calibration_curve_results_by_link_for_category_probabilities
            )
        )
        for link, sse in sse_by_link.items():
            sses_by_link[link].append(sse)
    return sses_by_link


def plot_calibration_curves_by_link_for_one_data_split(
    calibration_curve_results_by_link_for_category_probabilities: Dict[
        Link, CalibrationCurveResults
    ],
):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for (
        link_for_category_probabilities,
        calibration_curve_results,
    ) in calibration_curve_results_by_link_for_category_probabilities.items():
        ccr = calibration_curve_results
        ax.plot(
            ccr.mean_predicted_prob,
            ccr.fraction_of_positives,
            "s-",
            label="%s" % (link_for_category_probabilities.name,),
        )

    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title("Calibration plots  (reliability curve)")

    plt.tight_layout()
    plt.show()


def plot_calibration_curves_for_one_data_split_at_a_time(
    results_on_many_splits: List[ResultsOnOneSplit],
    n_bins: int,
):
    sses_by_link = (
        compute_sum_of_squared_calibration_errors_by_link_from_results_on_many_splits(
            results_on_many_splits, n_bins
        )
    )
    for i, results_on_one_split in enumerate(results_on_many_splits):
        print(
            f"On this split, calibration sses were {sses_by_link[Link.CBC_PROBIT][i]:.03} for CBC and {sses_by_link[Link.CBM_PROBIT][i]:.03} for CBM"
        )
        calibration_curve_results_by_link_for_category_probabilities = (
            get_calibration_curve_results_on_one_data_split(
                results_on_one_split, n_bins
            )
        )
        plot_calibration_curves_by_link_for_one_data_split(
            calibration_curve_results_by_link_for_category_probabilities
        )
        input("Press Enter to continue...")


def plot_CBM_calibration_advantage_curves(
    results_on_many_splits: List[ResultsOnOneSplit],
    n_bins: int,
):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    n_curves = len(results_on_many_splits)
    palette = sns.color_palette(None, n_curves)

    for (i, results_on_one_split) in enumerate(results_on_many_splits):
        calibration_curve_results_by_link_for_category_probabilities = (
            get_calibration_curve_results_on_one_data_split(
                results_on_one_split, n_bins
            )
        )
        rbl = calibration_curve_results_by_link_for_category_probabilities
        pred_CBC = rbl[Link.CBC_PROBIT].mean_predicted_prob
        pred_CBM = rbl[Link.CBM_PROBIT].mean_predicted_prob
        actual_CBC = rbl[Link.CBC_PROBIT].fraction_of_positives
        # actual_CBM = rbl[Link.CBC_PROBIT].fraction_of_positives
        actual = actual_CBC  # they are the same (actual_CBM=actual_CBC)
        overshoot_CBM = pred_CBM - actual
        overshoot_CBC = pred_CBC - actual
        advantage_is_positive = abs(overshoot_CBM) < abs(overshoot_CBC)
        sign = advantage_is_positive * 2 - 1
        advantage_CBM = abs(overshoot_CBC - overshoot_CBM) * sign

        # Merge locations which are identical
        xs = []
        ys = []
        ys_snip = []
        x_prev = -np.inf
        for (j, x) in enumerate(actual):
            ys_snip.append(advantage_CBM[j])
            if x != x_prev:
                xs.append(x)
                ys.append(np.mean(ys_snip))
                ys_snip = []
            x_prev = x

        ax.plot(xs, ys, color=palette[i], label=i)

    ax.set_xlabel("Bin, indexed by empirical probability of category")
    ax.set_ylabel("CBM calibration advantage")
    plt.legend(title="Data fold")
    plt.hlines(0, 0, 1, color="k")
    plt.tight_layout()
    plt.show()
