"""
Here we give different procedures for combining 
category probabilities from IB+CBC and IB+CBM 
"""
from enum import Enum

import numpy as np

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


class CombinationRule(Enum):
    """
    How to select or combine IB+CBC and IB+CBM models
    """

    IB_PLUS_CBC_IFF_SUM_OF_IB_PROBS_EXCEEDS_UNITY = 1
    IB_PLUS_CBC_IFF_PROB_DIAGONAL_ORTHANT_EXCEEDS_HALF = 2
    WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION = 3
    WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION_BUT_OVERRIDE_TO_ONLY_USE_IB_PLUS_CBM_WHEN_IB_ASSIGNS_LESS_THAN_HALF_OF_ITS_MASS_TO_THE_CBC_SET = (
        4
    )


def choose_IB_plus_CBC_iff_sum_of_IB_probs_exceeds_unity(
    CBC_predictions_with_IB_MLE: NumpyArray2D,
    CBM_predictions_with_IB_MLE: NumpyArray1D,
    sum_of_IB_probs: NumpyArray1D,
) -> NumpyArray2D:
    """
    Returns:
        New category probabilities that select, on a per sample basis,
        from the two input category probabilities.  This is
        an array of shape (N,K), where K is the number of categories.
        The n-th row gives a point on the simplex - a pmf over {1,...,K}.
    """

    # make combined predictions (need to convert to numpy arrays first)
    combined_predictions_with_IB_MLE = np.copy(CBC_predictions_with_IB_MLE)
    combined_predictions_with_IB_MLE[
        sum_of_IB_probs <= 1
    ] = CBM_predictions_with_IB_MLE[sum_of_IB_probs <= 1]
    return combined_predictions_with_IB_MLE


def choose_IB_plus_CBC_iff_prob_cbc_exceeds_half(
    CBC_predictions_with_IB_MLE: NumpyArray2D,
    CBM_predictions_with_IB_MLE: NumpyArray1D,
    log_Cs_CBC: NumpyArray1D,
) -> NumpyArray2D:
    """
    If probability mass in the set of diagonally opposed orthants where exactly one component is positive
    (i.e. where exactly one binary one-vs-rest classifier assigns the probability
    of success to be greater than .50) exceeds half, then we consider things predictable,
    and so we prefer CBC over CBM.

    Warning:
        It is highly recommended that you use `combine_plus_CBC_and_plus_CBM_via_cbc_probability`
        instead.   This is a similar function, but it weights the IB+CBM and IB+CBC category probabilities
        according to the probability mass in the CBC set, instead of making a hard selection.

        Why prefer the averaged approach?
        First off, the threshold whereby the probability mass contained in the CBC set
        "exceeds half" is arbitrary, and may not scale as K or other settings change.  Second, early
        empirical results suggest that the averaged approach does much better in reducing error without
        incurring a cost.

    Returns:
        New category probabilities that select, on a per sample basis,
        from the two input category probabilities.  This is
        an array of shape (N,K), where K is the number of categories.
        The n-th row gives a point on the simplex - a pmf over {1,...,K}.
    """
    # make combined predictions (need to convert to numpy arrays first)
    combined_predictions_with_IB_MLE = np.copy(CBM_predictions_with_IB_MLE)
    prob_cbc = np.exp(log_Cs_CBC)
    combined_predictions_with_IB_MLE[prob_cbc >= 0.5] = CBC_predictions_with_IB_MLE[
        prob_cbc >= 0.5
    ]
    return combined_predictions_with_IB_MLE


def weigh_ib_plus_do_via_probability_that_ib_is_in_do_region(
    CBC_predictions_with_IB_MLE: NumpyArray2D,
    CBM_predictions_with_IB_MLE: NumpyArray1D,
    log_Cs_CBC: NumpyArray1D,
):
    return _combine_plus_CBC_and_plus_CBM_via_cbc_probability(
        CBC_predictions_with_IB_MLE,
        CBM_predictions_with_IB_MLE,
        log_Cs_CBC,
        override_to_only_use_IB_plus_CBM_when_IB_assigns_less_than_half_of_its_mass_to_the_CBC_set=False,
    )


def weigh_ib_plus_do_via_probability_that_ib_is_in_do_region_but_override_to_only_use_ib_plus_sdo_when_ib_assigns_less_than_half_of_its_mass_to_the_do_set(
    CBC_predictions_with_IB_MLE: NumpyArray2D,
    CBM_predictions_with_IB_MLE: NumpyArray1D,
    log_Cs_CBC: NumpyArray1D,
):
    return _combine_plus_CBC_and_plus_CBM_via_cbc_probability(
        CBC_predictions_with_IB_MLE,
        CBM_predictions_with_IB_MLE,
        log_Cs_CBC,
        override_to_only_use_IB_plus_CBM_when_IB_assigns_less_than_half_of_its_mass_to_the_CBC_set=True,
    )


def _combine_plus_CBC_and_plus_CBM_via_cbc_probability(
    CBC_predictions_with_IB_MLE: NumpyArray2D,
    CBM_predictions_with_IB_MLE: NumpyArray1D,
    log_Cs_CBC: NumpyArray1D,
    override_to_only_use_IB_plus_CBM_when_IB_assigns_less_than_half_of_its_mass_to_the_CBC_set: bool,
) -> NumpyArray2D:
    """
    The higher the probability mass that the IB model assigns to the set of
    diagonally opposed orthants where exactly one component is positive,
    (i.e. where exactly one binary one-vs-rest classifier assigns the probability
    of success to be greater than .50), the more predictable things are, and the more we prefer CBC.

    Returns:
        New category probabilities that select, on a per sample basis,
        from the two input category probabilities.  This is
        an array of shape (N,K), where K is the number of categories.
        The n-th row gives a point on the simplex - a pmf over {1,...,K}.
    """
    # make combined predictions (need to convert to numpy arrays first)
    prob_cbc = np.exp(log_Cs_CBC)
    combined_predictions = (
        prob_cbc[:, np.newaxis] * CBC_predictions_with_IB_MLE
        + (1 - prob_cbc[:, np.newaxis]) * CBM_predictions_with_IB_MLE
    )
    if override_to_only_use_IB_plus_CBM_when_IB_assigns_less_than_half_of_its_mass_to_the_CBC_set:
        combined_predictions[prob_cbc <= 0.5] = CBM_predictions_with_IB_MLE[
            prob_cbc <= 0.5
        ]
    return combined_predictions


COMBINATION_FUNCTION_BY_COMBINATION_RULE = {
    CombinationRule.IB_PLUS_CBC_IFF_SUM_OF_IB_PROBS_EXCEEDS_UNITY: choose_IB_plus_CBC_iff_sum_of_IB_probs_exceeds_unity,
    CombinationRule.IB_PLUS_CBC_IFF_PROB_DIAGONAL_ORTHANT_EXCEEDS_HALF: choose_IB_plus_CBC_iff_prob_cbc_exceeds_half,
    CombinationRule.WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION: weigh_ib_plus_do_via_probability_that_ib_is_in_do_region,
    CombinationRule.WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION_BUT_OVERRIDE_TO_ONLY_USE_IB_PLUS_CBM_WHEN_IB_ASSIGNS_LESS_THAN_HALF_OF_ITS_MASS_TO_THE_CBC_SET: weigh_ib_plus_do_via_probability_that_ib_is_in_do_region_but_override_to_only_use_ib_plus_sdo_when_ib_assigns_less_than_half_of_its_mass_to_the_do_set,
}


def construct_combined_category_probabilities(
    CBC_predictions_with_IB_MLE: NumpyArray2D,
    CBM_predictions_with_IB_MLE: NumpyArray1D,
    log_Cs_CBC: NumpyArray1D,
    sum_of_IB_probs: NumpyArray1D,
    combination_rule: CombinationRule,
):
    if (
        combination_rule
        == CombinationRule.IB_PLUS_CBC_IFF_SUM_OF_IB_PROBS_EXCEEDS_UNITY
    ):
        combination_basis = sum_of_IB_probs
    elif (
        combination_rule
        == CombinationRule.IB_PLUS_CBC_IFF_SUM_OF_IB_PROBS_EXCEEDS_UNITY
        or combination_rule
        == CombinationRule.WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION
        or combination_rule
        == CombinationRule.WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION_BUT_OVERRIDE_TO_ONLY_USE_IB_PLUS_CBM_WHEN_IB_ASSIGNS_LESS_THAN_HALF_OF_ITS_MASS_TO_THE_CBC_SET
    ):
        combination_basis = log_Cs_CBC
    else:
        raise ValueError(f"I don't understand the combination rule {combination_rule}")

    function_to_combine_category_probabilities = (
        COMBINATION_FUNCTION_BY_COMBINATION_RULE[combination_rule]
    )
    return function_to_combine_category_probabilities(
        CBC_predictions_with_IB_MLE, CBM_predictions_with_IB_MLE, combination_basis
    )
