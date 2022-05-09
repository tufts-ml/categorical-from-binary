import warnings
from typing import List, NamedTuple, Optional

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    HierarchicalMulticlassRegressionDataset,
)
from categorical_from_binary.ib_cavi.multi.hierarchical_ib_probit.inference import (
    VariationalParams,
    data_dimensions_from_hierarchical_dataset,
)


class HierarchicalMetrics(NamedTuple):
    mean_choice_probs_by_group: List[float]
    mean_choice_probs_by_group_for_rarest_category: Optional[List[float]]


def find_rarest_category(
    data: HierarchicalMulticlassRegressionDataset,
) -> int:
    category_counts_per_sequence = [sum(dataset.labels, 0) for dataset in data.datasets]
    rarest_category = np.argmin(np.mean(category_counts_per_sequence, 0))
    return rarest_category


def compute_metrics_for_hierarchical_multiclass_regression(
    variational_params: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
    rarest_category: Optional[int] = None,
) -> HierarchicalMetrics:
    """
    Arguments:
        rarest_category:
            Optional.  If not provided, computed from the dataset.
    """

    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # Ns_are_all_equal=Ns[1:]==Ns[:-1]
    # if not Ns_are_all_equal:
    #     raise NotImplementedError(f"Currently assuming all sequences are of the same"
    #     f"length just out of laziness, so that I can use matrices instead of lists")
    # N=Ns[0]

    choice_probs_by_group = compute_choice_probs_for_hierarchical_multiclass_regression(
        variational_params, data
    )
    mean_choice_probs_by_group = [
        np.mean(choice_probs) for choice_probs in choice_probs_by_group
    ]

    if rarest_category is None:
        rarest_category = find_rarest_category(data)
    choice_probs_by_group_for_rarest_category = (
        compute_choice_probs_for_hierarchical_multiclass_regression(
            variational_params, data, focal_choice=rarest_category
        )
    )
    mean_choice_probs_by_group_for_rarest_category = [
        np.nanmean(choice_probs)
        for choice_probs in choice_probs_by_group_for_rarest_category
    ]
    return HierarchicalMetrics(
        mean_choice_probs_by_group,
        mean_choice_probs_by_group_for_rarest_category,
    )


def compute_mean_choice_probs_for_hierarchical_multiclass_regression(
    variational_params: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
) -> List[List[float]]:
    choice_probs_by_group = compute_choice_probs_for_hierarchical_multiclass_regression(
        variational_params, data
    )
    return [np.mean(choice_probs) for choice_probs in choice_probs_by_group]


def compute_mean_log_choice_probs_for_hierarchical_multiclass_regression(
    variational_params: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
) -> List[List[float]]:
    choice_probs_by_group = compute_choice_probs_for_hierarchical_multiclass_regression(
        variational_params, data
    )
    return [np.mean(np.log(choice_probs)) for choice_probs in choice_probs_by_group]


def compute_choice_probs_for_hierarchical_multiclass_regression(
    variational_params: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
    focal_choice: Optional[int] = None,
) -> List[List[float]]:
    """
    Arguments:
        focal_choice:
            Optional.  If present, only return the choice probs when the label was
            the `focal_choice`.  Otherwise, return all choice probs.
    Returns:
        A List of Lists of floats.  The outer list is indexed by group.
        The inner list is the probability assigned by the model to categorical
        observations for that group.  By default, all such probabilities are returned.
        If `focal_choice` is provided, then a restricted set of choice probabilities are
        provided.
    """

    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    choice_probs = [None] * J
    for j in range(J):
        labels_j = data.datasets[j].labels
        choices_j = np.argmax(labels_j, 1)  # index of selected category

        features_j = data.datasets[j].features
        beta_expected_j = variational_params.beta_means[j, :, :]

        category_probs_j = construct_category_probs(
            features_j, beta_expected_j, Link.CBM_PROBIT
        )
        choice_probs_j = [
            category_probs_j[i, choice] for (i, choice) in enumerate(choices_j)
        ]
        if focal_choice is not None:
            choice_probs_j_filtered = list(
                np.array(choice_probs_j)[choices_j == focal_choice]
            )
            choice_probs[j] = choice_probs_j_filtered
        else:
            choice_probs[j] = choice_probs_j
    return choice_probs


def evaluate_variational_model_for_hierarchical_multiclass_regression(
    variational_params: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
):
    warnings.warn(
        f"We are currently using the expected beta instead of the posterior predictive "
        f"when computing category probabilities with the variational approximation."
    )

    # TODO: Improve this to use the posterior predictive.

    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    choice_probs_all = []

    print(
        f"\n\nNum groups: {J}, mean number of observations per group: {np.mean(Ns):.03}, "
        f"num categories: {K}, num designed features: {M} \n"
    )

    for j in range(J):
        labels_j = data.datasets[j].labels
        choices_j = np.argmax(labels_j, 1)  # index of selected category

        features_j = data.datasets[j].features
        beta_expected_j = variational_params.beta_means[j, :, :]

        category_probs_j = construct_category_probs(
            features_j, beta_expected_j, Link.CBC_PROBIT
        )
        choice_probs_j = [
            category_probs_j[i, choice] for (i, choice) in enumerate(choices_j)
        ]
        print(
            f"The variational model's mean choice probability for group {j}/{J} "
            f"is {np.mean(choice_probs_j):.03}"
        )

        choice_probs_all.extend(choice_probs_j)

    print(
        f"\nThe variational model's overall mean choice probability (across groups) "
        f"is {np.mean(choice_probs_all):.03}"
    )
