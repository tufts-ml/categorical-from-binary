"""
Variational Inference for Bayesian multiclass logistic regression with 
* stick-breaking link function
* polya-gamma augmentation (for conditional conjugacy)

There are currently problems with inference, as shown in the demo.
"""

import copy
from typing import List, NamedTuple, Optional

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    MulticlassRegressionDataset,
)
from categorical_from_binary.polya_gamma.binary_logreg_vi.inference import (
    compute_expected_value_of_c_parameter_for_polya_gamma,
)
from categorical_from_binary.polya_gamma.polya_gamma import compute_polya_gamma_expectation
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


class VariationalParameters_MulticlassLogisticRegression(NamedTuple):
    means_beta: List[NumpyArray1D]
    covs_beta: List[NumpyArray2D]


class PriorParameters_MulticlassLogisticRegression(NamedTuple):
    means_beta: List[NumpyArray1D]
    covs_beta: List[NumpyArray2D]


def compute_num_trials(labels: NumpyArray2D) -> NumpyArray2D:
    """
    Computes the "number of trials" associated with a given sample and category,
    needed for the stick-breaking representation of a categorical distribution.

    This value is equal to N_{ik} in the Linderman paper,  E_{ik} in one of my reports
    (with notation as of 3/13/2021), and b_{ik} when doing variational inference.

    It is simply equal to 1 minus the sum of labels from previous categories.
    """
    ## Below is my previous code.
    cumsum_of_labels = np.cumsum(labels, 1)
    sum_of_labels_from_earlier_categories = np.insert(
        cumsum_of_labels, 0, values=0.0, axis=1
    )[:, :-1]
    return 1 - sum_of_labels_from_earlier_categories


def compute_kappa(labels: NumpyArray2D, num_trials: NumpyArray2D) -> NumpyArray2D:
    """
    In the multiclass logistic regression setting,  kappa is simply
        * -1/2 for a category that was not selected for that sample (but an upcoming category was)
        * 1/2 for a category that was selected for that sample
        * 0 for a category that was not selected, but an earlier category was.

    Arguments:
        labels: np.array with shape (n_samples, n_categories)
    Returns:
        np.array with shape (n_samples, n_categories)
    """
    # TODO: Add unit test that coverts labels to kappa in the expected way:
    #   If k is the selected category for observation i, then
    #       * kappa[i,k] = 0.5.
    #       * For j<k, kappa[i,j] = -0.5.
    #       * For l>k, kappa[i,l] = 0.0
    return labels - num_trials / 2


###
# Main inference routine
###


def run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
    dataset: MulticlassRegressionDataset,
    prior_params: PriorParameters_MulticlassLogisticRegression,
    variational_params_init: Optional[
        VariationalParameters_MulticlassLogisticRegression
    ],
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_elbo: float = -np.inf,
    verbose: bool = False,
) -> VariationalParameters_MulticlassLogisticRegression:
    """
    Use variational inference with polya gamma augementation to approximate the posterior mean and covariance
    on the regression weights of a Bayesian logistic regression

    Arguments:
        prior_params: prior mean and covariance for the regression weights
        variational_params_init: If not provided, we will use the prior params to initialize.
    """
    if max_n_iterations == np.inf and convergence_criterion_drop_in_elbo == -np.inf:
        raise ValueError(
            f"You must change max_n_iterations and/or convergence_criterion_drop_in_elbo "
            f"from the default value so that the algorithm knows when to stop"
        )

    if convergence_criterion_drop_in_elbo != -np.inf:
        raise NotImplementedError(
            "I haven't yet implemented the ELBO for this model; it's a TODO"
        )

    # initialization
    prior_means_by_category, prior_covs_by_category = prior_params
    prior_precisions = [np.linalg.inv(cov) for cov in prior_covs_by_category]
    features = dataset.features
    features_transposed = np.transpose(features)
    num_trials = compute_num_trials(dataset.labels)
    n_samples, n_categories = np.shape(dataset.labels)

    kappa = compute_kappa(dataset.labels, num_trials)

    # initialization for top-level variational parameters
    if variational_params_init is not None:
        variational_mean_for_betas = variational_params_init.means_beta
        variational_cov_for_betas = variational_params_init.covs_beta
    else:
        # TODO: Use Linderman Default which accounts for label asymmetry
        variational_mean_for_betas = copy.copy(prior_means_by_category)
        variational_cov_for_betas = copy.copy(prior_covs_by_category)

    n_iterations_so_far = 0
    print(
        f"Max # iterations: {max_n_iterations}.  Convergence criterion (drop in ELBO): {convergence_criterion_drop_in_elbo}"
    ) if verbose else None
    print(f"\nTrue beta: {dataset.beta}\n") if verbose else None
    print(
        f"At iteration 0, the variational means by category are: {variational_mean_for_betas}"
    ) if verbose else None

    drop_in_elbo = np.inf

    while (
        n_iterations_so_far <= max_n_iterations
        and drop_in_elbo >= convergence_criterion_drop_in_elbo
    ):
        # TODO: vectorize this loop
        for k in range(n_categories - 1):
            expected_c_parameters_for_this_k = (
                compute_expected_value_of_c_parameter_for_polya_gamma(
                    features,
                    variational_mean_for_betas[k],
                    variational_cov_for_betas[k],
                )
            )
            polya_gamma_expectations_for_this_k = np.array(
                [
                    compute_polya_gamma_expectation(
                        num_trials[i, k], c, allow_b_to_be_zero=True
                    )
                    for (i, c) in enumerate(expected_c_parameters_for_this_k)
                ]
            )
            # TODO: the X'X might be very intensive with many samples and few covariates.  could perhaps use random projection.
            variational_cov_for_betas[k] = np.linalg.inv(
                features_transposed
                @ (polya_gamma_expectations_for_this_k[:, np.newaxis] * features)
                + prior_precisions[k]
            )
            variational_mean_for_betas[k] = variational_cov_for_betas[k] @ (
                features_transposed @ kappa[:, k]
                + prior_precisions[k] @ prior_means_by_category[k]
            )
        variational_params = VariationalParameters_MulticlassLogisticRegression(
            variational_mean_for_betas, variational_cov_for_betas
        )
        # TODO: Add ELBO BACK IN
        n_iterations_so_far += 1
        print(
            f"At iteration {n_iterations_so_far}, the variational means by category are: {variational_params.means_beta}"
        ) if verbose else None
        #
    return variational_params
