"""
Why?

1. Can get MLE of CBM and MLE of IB to see if they line up as n->infty 
2. Want to get variance in a category prob as a function of the variance in beta.

"""
import jax.numpy as np


np.set_printoptions(precision=3, suppress=True)

import collections
from typing import List, Optional

import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions as compute_CBC_Probit_predictions_with_autodiff_MLE,
    compute_CBM_Probit_predictions as compute_CBM_Probit_predictions_with_autodiff_MLE,
    compute_softmax_predictions,
    optimize_beta_for_CBC_model,
    optimize_beta_for_CBM_model,
    optimize_beta_for_IB_model,
    optimize_beta_for_softmax_model,
)
from categorical_from_binary.autodiff.simplex_distances import compute_mean_l1_distance
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_cbc_probit_probabilities,
    construct_cbm_probit_probabilities,
    construct_softmax_probabilities,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.timing import time_me


def run_warping_simulations_on_softmax_data(
    Ks: List[int],
    Ns: List[int],
    Ms: List[int],
    n_datasets_per_data_dimension: int,
    run_softmax: bool = True,
    run_sdo_and_do_mle: bool = True,
    run_sdo_and_do_mle_only_if_num_coefficients_is_less_than_this: Optional[
        float
    ] = None,
    run_cavi: bool = False,
    run_cavi_only_for_one_seed: bool = False,
    verbose_cavi=False,
    max_n_cavi_iterations: float = np.inf,
    convergence_criterion_drop_in_mean_elbo: float = -np.inf,
) -> DataFrame:
    """
    Generate data from softmax with given specifications (N samples, K categories, M covariates); compute the MLE of
    beta from the IB model (which is a proxy for the posterior mean beta from IB-CAVI); and then compare the distance
    in category probabilities to the ground truth.

    Note:
        This function is older than the one in simulations.core.performance; perhaps see that function first for a
        possibly better structure.
    """

    SKIPPED_FLAG = "skipped"

    d = collections.defaultdict(list)
    # `collections.defaultdict(list)` lets me append to lists that aren't there yet by
    #   d["my list"].append(new_item)
    #

    for K in Ks:
        for M in Ms:
            for N in Ns:
                seeds = [i for i in range(n_datasets_per_data_dimension)]
                for seed in seeds:

                    if run_cavi_only_for_one_seed:
                        run_cavi = seed == 0

                    if (
                        run_sdo_and_do_mle_only_if_num_coefficients_is_less_than_this
                        is not None
                    ):
                        run_sdo_and_do_mle = (
                            K * M
                            < run_sdo_and_do_mle_only_if_num_coefficients_is_less_than_this
                        )

                    print(f"\n --- Now analyzing K={K}, M={M}, N={N}, seed={seed} --- ")
                    d["K"].append(K)
                    d["M"].append(M)
                    d["N"].append(N)
                    d["seed"].append(seed)

                    ###
                    # Construct dataset
                    ###
                    n_categories = K
                    n_features = M
                    n_samples = N
                    include_intercept = True
                    link = Link.MULTI_LOGIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
                    dataset = generate_multiclass_regression_dataset(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_categories=n_categories,
                        beta_0=None,
                        link=link,
                        seed=seed,
                        include_intercept=include_intercept,
                    )

                    beta_init = np.zeros((n_features + 1, n_categories))

                    beta_star_IB_MLE, time_for_IB_MLE = time_me(
                        optimize_beta_for_IB_model
                    )(
                        beta_init,
                        dataset.features,
                        dataset.labels,
                        verbose=verbose_cavi,
                    )

                    category_probs_CBC_with_IB_MLE = construct_cbc_probit_probabilities(
                        dataset.features, beta_star_IB_MLE
                    )

                    category_probs_CBM_with_IB_MLE = construct_cbm_probit_probabilities(
                        dataset.features, beta_star_IB_MLE
                    )

                    category_probs_true = construct_softmax_probabilities(
                        dataset.features, dataset.beta
                    )

                    # Comparing IB beta to CBM/CBC beta is hard becuase of non-identifiability.  Very different betas
                    # can give essentially the same category probs.  An example:  beta_star_IB_MLE and beta_star_CBM_MLE
                    # are very different, but they give essentially the same CBM category probs.
                    #
                    # To get around this problem while still comparing CBM to CBC, we could perhaps propose that
                    # Difference_in_some_sense( CBM_probs(beta_star_IB_MLE), CBM_probs(beta_star_CBM_MLE) )  is less than
                    # Difference_in_some_sense( CBC_probs(beta_star_IB_MLE), CBC_probs(beta_star_CBC_MLE) )

                    error_to_ground_truth_probs_when_using_IB_MLE_plus_CBC = (
                        compute_mean_l1_distance(
                            category_probs_CBC_with_IB_MLE, category_probs_true
                        )
                    )
                    error_to_ground_truth_probs_when_using_IB_MLE_plus_CBM = (
                        compute_mean_l1_distance(
                            category_probs_CBM_with_IB_MLE, category_probs_true
                        )
                    )
                    print(
                        f"Error compared to ground truth. IB MLE + CBM: {error_to_ground_truth_probs_when_using_IB_MLE_plus_CBM :.03f}, IB MLE + CBC: {error_to_ground_truth_probs_when_using_IB_MLE_plus_CBC :.03f}"
                    )
                    # TODO: show category probs for CAVI -> category probs for MLE as n to infinity.  just for funsies / sanity check.

                    d["error with IB MLE and CBM"].append(
                        error_to_ground_truth_probs_when_using_IB_MLE_plus_CBM
                    )
                    d["error with IB MLE and CBC"].append(
                        error_to_ground_truth_probs_when_using_IB_MLE_plus_CBC
                    )
                    d["time for IB MLE"].append(time_for_IB_MLE)

                    if run_sdo_and_do_mle:
                        beta_star_CBC_MLE, time_for_CBC_MLE = time_me(
                            optimize_beta_for_CBC_model
                        )(
                            beta_init,
                            dataset.features,
                            dataset.labels,
                            verbose=False,
                        )
                        category_probs_with_CBC_MLE = (
                            compute_CBC_Probit_predictions_with_autodiff_MLE(
                                beta_star_CBC_MLE, dataset.features
                            )
                        )

                        error_to_ground_truth_probs_when_using_CBC_MLE = (
                            compute_mean_l1_distance(
                                category_probs_with_CBC_MLE, category_probs_true
                            )
                        )

                        beta_star_CBM_MLE, time_for_CBM_MLE = time_me(
                            optimize_beta_for_CBM_model
                        )(
                            beta_init,
                            dataset.features,
                            dataset.labels,
                            verbose=False,
                        )
                        category_probs_with_CBM_MLE = (
                            compute_CBM_Probit_predictions_with_autodiff_MLE(
                                beta_star_CBM_MLE, dataset.features
                            )
                        )

                        error_to_ground_truth_probs_when_using_CBM_MLE = (
                            compute_mean_l1_distance(
                                category_probs_with_CBM_MLE, category_probs_true
                            )
                        )

                        print(
                            f"Error compared to ground truth. CBM MLE: {error_to_ground_truth_probs_when_using_CBM_MLE :.03f}, CBC MLE: {error_to_ground_truth_probs_when_using_CBC_MLE :.03f}"
                        )

                        d["error with CBM MLE"].append(
                            error_to_ground_truth_probs_when_using_CBM_MLE
                        )
                        d["error with CBC MLE"].append(
                            error_to_ground_truth_probs_when_using_CBC_MLE
                        )
                        d["time for CBM MLE"].append(time_for_CBM_MLE)
                        d["time for CBC MLE"].append(time_for_CBC_MLE)
                    else:
                        d["error with CBM MLE"].append(SKIPPED_FLAG)
                        d["error with CBC MLE"].append(SKIPPED_FLAG)
                        d["time for CBM MLE"].append(SKIPPED_FLAG)
                        d["time for CBC MLE"].append(SKIPPED_FLAG)
                    if run_softmax:
                        beta_star_softmax_MLE, time_for_softmax_MLE = time_me(
                            optimize_beta_for_softmax_model
                        )(
                            beta_init,
                            dataset.features,
                            dataset.labels,
                            verbose=False,
                        )
                        category_probs_with_softmax_MLE = compute_softmax_predictions(
                            beta_star_softmax_MLE,
                            dataset.features,
                        )

                        error_to_ground_truth_probs_when_using_softmax_MLE = (
                            compute_mean_l1_distance(
                                category_probs_with_softmax_MLE, category_probs_true
                            )
                        )
                        print(
                            f"Error compared to ground truth. softmax MLE: {error_to_ground_truth_probs_when_using_softmax_MLE :.03f}"
                        )
                        d["error with softmax MLE"].append(
                            error_to_ground_truth_probs_when_using_softmax_MLE
                        )
                        d["time for softmax MLE"].append(time_for_softmax_MLE)
                    else:
                        d["error with softmax MLE"].append(SKIPPED_FLAG)
                        d["time for softmax MLE"].append(SKIPPED_FLAG)
                    if run_cavi:
                        results, time_for_IB_CAVI = time_me(
                            compute_multiclass_probit_vi_with_normal_prior
                        )(
                            dataset.labels,
                            dataset.features,
                            variational_params_init=None,
                            convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
                            max_n_iterations=max_n_cavi_iterations,
                            verbose=False,
                        )
                        beta_CAVI_mean = results.variational_params.beta.mean

                        category_probs_CBC_with_IB_CAVI_mean = (
                            construct_cbc_probit_probabilities(
                                dataset.features, beta_CAVI_mean
                            )
                        )

                        category_probs_CBM_with_IB_CAVI_mean = (
                            construct_cbm_probit_probabilities(
                                dataset.features, beta_CAVI_mean
                            )
                        )
                        error_to_ground_truth_probs_when_using_CBM_variational_posterior_mean = compute_mean_l1_distance(
                            category_probs_CBM_with_IB_CAVI_mean, category_probs_true
                        )
                        error_to_ground_truth_probs_when_using_CBC_variational_posterior_mean = compute_mean_l1_distance(
                            category_probs_CBC_with_IB_CAVI_mean, category_probs_true
                        )
                        print(
                            f"Error compared to ground truth. IB CAVI mean + CBM: {error_to_ground_truth_probs_when_using_CBM_variational_posterior_mean :.03f}, IB CAVI mean + CBC: {error_to_ground_truth_probs_when_using_CBC_variational_posterior_mean :.03f}"
                        )
                        d["error with IB CAVI mean and CBM"].append(
                            error_to_ground_truth_probs_when_using_CBM_variational_posterior_mean
                        )
                        d["error with IB CAVI mean and CBC"].append(
                            error_to_ground_truth_probs_when_using_CBC_variational_posterior_mean
                        )
                        d["time for IB CAVI"].append(time_for_IB_CAVI)
                    else:
                        d["error with IB CAVI mean and CBM"].append(SKIPPED_FLAG)
                        d["error with IB CAVI mean and CBC"].append(SKIPPED_FLAG)
                        d["time for IB CAVI"].append(SKIPPED_FLAG)
                    print(
                        f'Running times: IB-MLE: {d["time for IB MLE"][-1]}, '
                        f'softmax MLE: {d["time for softmax MLE"][-1]}, '
                        f'CBM MLE: {d["time for CBM MLE"][-1]}, '
                        f'CBC MLE: {d["time for CBC MLE"][-1]}, '
                        f'IB CAVI: {d["time for IB CAVI"][-1]}, '
                    )
    return pd.DataFrame(d)
