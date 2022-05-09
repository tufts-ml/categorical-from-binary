"""
Here we do coordinate ascent variational inference (CAVI)
on the IB model (Wojnowicz et al 2021); the resulting distribution on 
betas can be used within the CBC model (Johndrow et al 2013)
or the CBM model (Wojnowicz et al 2021)
"""

import collections
import sys
import time
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import spmatrix

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.ib_cavi.multi.design_matrix import construct_design_matrix
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.shrinkage_groups import (
    ShrinkageGroupingStrategy,
)
from categorical_from_binary.ib_cavi.multi.structs import (
    CAVI_Results,
    ELBO_Stats,
    Precomputables,
    PriorType,
    VariationalParams,
)
from categorical_from_binary.performance_over_time.results import (
    get_most_recent_performance_results_as_string,
    update_performance_results,
)
from categorical_from_binary.selection.hyperparameters import Hyperparameters
from categorical_from_binary.types import NumpyArray2D


def compute_precomputables_dummy(
    variational_params_init: VariationalParams,
    design_matrix: NumpyArray2D,
    prior_type: PriorType = PriorType.NORMAL,
    shrinkage_grouping_strategy: Optional[ShrinkageGroupingStrategy] = None,
) -> Precomputables:
    return Precomputables(None, None, None)


def compute_multiclass_vi(
    labels: Union[NumpyArray2D, spmatrix],
    covariates: Union[NumpyArray2D, spmatrix],
    links_for_category_probabilities: List[Link],
    initialize_variational_params: Callable,
    update_variational_params_and_elbo_stats: Callable,
    compute_precomputables: Callable,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_mean_elbo: float = -np.inf,
    use_autoregressive_design_matrix: bool = False,
    labels_test: Optional[NumpyArray2D] = None,
    covariates_test: Optional[NumpyArray2D] = None,
    prior_beta_mean: Optional[NumpyArray2D] = None,
    prior_beta_precision: Optional[NumpyArray2D] = None,
    variational_params_init: Optional[VariationalParams] = None,
    prior_type: PriorType = PriorType.NORMAL,
    hyperparameters: Optional[Hyperparameters] = None,
    shrinkage_grouping_strategy: Optional[ShrinkageGroupingStrategy] = None,
    verbose: bool = True,
) -> CAVI_Results:
    """
    A general inference wrapper for doing coordinate ascent variational inference with
    categorical models.  The inference is done on an
    independent binary model (Wojnowicz et al 2021), and so one can choose different category probability
    formulas within which to plug in the resulting betas.  The initial release of this was used by (S)CBC-Probit and CBC-Logit
    with polya gamma augmentation.

    Arguments:
        labels: array with shape (n_obs, n_categories)
            one-hot encoded representations of response categories
        covariates: array with shape (n_obs, n_covariates)
            In the autoregressive case, the covariates are only the exogenous covariates;
            the code automatically constructs the previous category as a feature.
        convergence_criterion_drop_in_mean_elbo:  We base our convergence criterion on mean ELBO not ELBO!
            For example, a drop of 1.0 in the ELBO would be HUGE for a small dataset but MINISCULE
            for a large dataset.  In particular we divide the elbo by NxK, where N is the number of samples and K
            is the number of categories. So this specification should at least make a good value independent of
            sample size and number of categories.
        use_autoregressive_design_matrix: bool
            If true, the response categories (i.e. labels) are taken to be a sequence, and the model uses
            the previous response category as a predictor of the upcoming response category.
        labels_test : Optional
            If present, we will compute holdout likelihood over time
        covariates_test : Optional
            If present, we will compute holdout likelihood over time
        beta_variational_mean_init: Optional
            array with shape (beta_dim, )
            where, if the covariates contain an intercept term,
                beta_dim = n_covariates (in the non-autoregressive case)
                         = (n_covariates - 1) + n_categories (in the AR case, if the covariates contain an intercept term)
                         = n_covariates + n_categories (in the AR case, if the covariates don't contain an intercept term)
        hyperparameters:  Optional, does not need to be specified if prior_type is NORMAL.
            does need to be specified if prior_type is NORMAL_GAMMA
        shrinkage_grouping_strategy: Optional, does not need to be specified if prior_type is NORMAL.
            does need to be specified if prior_type is NORMAL_GAMMA
    Returns:
        CAVI_Results. Includes VariationalParameters.  Which in turn includes:
            beta_mean, beta_cov: The variational parameters for the normal distribution.
                beta_mean is an array with shape (beta_dim, n_categories)
                beta_cov is an array with shape (beta_dim, beta_dim)
                    note that the covariance is the same for each category, so
                    we just return a single beta_cov instead of an array
                    of them with shape (n_categories, )
    Notes:
        - (On sparse processing):
            If we pass in a sparse `covariates` matrix (X), and prior_type is NORMAL,
            we notice this and do a bunch of sparse computations along the way, including but not necessarily limited to:
                * We computing XtX in a sparse manner
                * We compute cov(beta)=(I+XtX)^{-1} in a sparse manner
                * We compute E_q[Z] in a sparse manner
            all of which can dramatically speed up the up-front computations when matrices are large and sparse, and can
            save on memory costs.  Note that we should extend this to the category probability computation and the ELBO.

            We also sparsify the design matrix on the test set data, which
            can speed up computations of category probabilities due to speeding up of the computation of the linear
            predictor X_test'beta.

    """
    if prior_beta_mean is not None or prior_beta_precision is not None:
        raise NotImplementedError(
            "Some computations may still currently assume the prior on beta is N(0,I). Need to double check function body."
        )

    if prior_type == PriorType.NORMAL:
        warnings.warn(
            "We currently assume the prior on beta is N(0,I). Does that mean and variance make sense?"
        )

    if prior_type == PriorType.NORMAL_GAMMA and hyperparameters is None:
        raise ValueError(
            "If prior type is NORMAL_GAMMA, you must provide hyperparameters."
        )

    if (
        max_n_iterations == np.inf
        and convergence_criterion_drop_in_mean_elbo == -np.inf
    ):
        raise ValueError(
            f"You must change max_n_iterations and/or convergence_criterion_drop_in_elbo "
            f"from the default value so that the algorithm knows when to stop"
        )

    if convergence_criterion_drop_in_mean_elbo != -np.inf and scipy.sparse.issparse(
        covariates
    ):
        raise NotImplementedError(
            f"You've passed in a sparse `covariates` matrix, so the function will do sparse computations along the way. "
            f"But the ELBO code is not currently constructed in such a way so as to be able to handle sparse matrices. "
            f"Would it work to use `max_n_iterations` as a stopping criterion instead? If not, you'll probably need "
            f"to do some code development."
        )

    if (
        prior_type == PriorType.NORMAL_GAMMA
        and convergence_criterion_drop_in_mean_elbo != -np.inf
    ):
        raise ValueError(
            "We do not currently have a way to compute the ELBO under the normal-gamma prior."
        )

    ### Start timer
    start_time_for_up_front_computations = time.time()

    ### Prepare training data
    design_matrix = construct_design_matrix(
        covariates,
        labels,
        use_autoregressive_design_matrix,
    )

    # initalize
    n_samples, n_categories = np.shape(labels)
    variational_params = initialize_variational_params(
        design_matrix,
        n_categories,
        variational_params_init,
    )
    precomputables = compute_precomputables(
        variational_params, design_matrix, prior_type, shrinkage_grouping_strategy
    )

    ### Report time for up-front computations
    end_time_for_up_front_computations = time.time()
    elapsed_secs_for_up_front_computations = (
        end_time_for_up_front_computations - start_time_for_up_front_computations
    )

    if verbose:
        print(
            f"\nMax # iterations: {max_n_iterations}.  Convergence criterion (drop in mean ELBO): {convergence_criterion_drop_in_mean_elbo}",
            end="\n",
        )
        print(
            f"Elapsed time (secs) for up front computations: {elapsed_secs_for_up_front_computations:.3f}",
            end="\n\n",
        )

    elapsed_secs_for_cavi_iterations = 0.0
    elapsed_secs_for_computing_category_probabilities = 0.0

    ### Prepare holdout performance computation (if using)
    performance_over_time_as_dict = collections.defaultdict(list)
    compute_performance_over_time = (
        labels_test is not None and covariates_test is not None
    )
    # `TODO` Right now, we assume that we `compute_performance_over_time` if test data is passed in.
    # But we might want to at least compute performance over time on training data only, which is always
    # possible.  Fix this up by making `compute_performance_over_time` a boolean, and update the logic
    # accordingly.
    if compute_performance_over_time:

        # TODO: The construction of autoregressive design matrix should be done OUTSIDE
        # of the inference function.

        design_matrix_train = construct_design_matrix(
            covariates,
            labels,
            use_autoregressive_design_matrix,
        )
        design_matrix_test = construct_design_matrix(
            covariates_test,
            labels_test,
            use_autoregressive_design_matrix,
        )

        for (l, link) in enumerate(links_for_category_probabilities):
            update_performance_results(
                performance_over_time_as_dict,
                covariates_train=design_matrix_train,
                labels_train=labels,
                beta_mean=variational_params.beta.mean,
                secs_elapsed=elapsed_secs_for_cavi_iterations,
                link=link,
                covariates_test=covariates_test,
                labels_test=labels_test,
                update_secs_elapsed=l,
            )
        performance_over_time_as_dict["ELBO (mean over N,K)"].append(-np.inf)
        performance_over_time_as_dict["seconds elapsed (category probs)"].append(
            elapsed_secs_for_computing_category_probabilities
        )

    ### Inference
    n_iterations_so_far = 0
    elbo_stats = ELBO_Stats(
        convergence_criterion_drop_in_mean_elbo,
        previous_mean_elbo=-np.inf,
        mean_elbo=-np.inf,
        drop_in_mean_elbo=np.inf,
    )

    while (
        n_iterations_so_far < max_n_iterations
        and elbo_stats.drop_in_mean_elbo
        >= elbo_stats.convergence_criterion_drop_in_mean_elbo
    ):
        start_time_for_this_cavi_iteration = time.time()

        variational_params, elbo_stats = update_variational_params_and_elbo_stats(
            variational_params,
            elbo_stats,
            design_matrix,
            labels,
            prior_beta_mean,
            prior_beta_precision,
            precomputables,
            hyperparameters,
            prior_type,
        )

        n_iterations_so_far += 1

        end_time_for_this_cavi_iteration = time.time()
        elapsed_secs_for_cavi_iterations += (
            end_time_for_this_cavi_iteration - start_time_for_this_cavi_iteration
        )

        ### Compute performance metrics on train and test set
        if compute_performance_over_time:
            start_time_for_computing_these_cat_probs = time.time()

            # TODO: The construction of autoregressive design matrix should be done OUTSIDE
            # of the inference function.
            design_matrix_train = construct_design_matrix(
                covariates,
                labels,
                use_autoregressive_design_matrix,
            )
            design_matrix_test = construct_design_matrix(
                covariates_test,
                labels_test,
                use_autoregressive_design_matrix,
            )

            secs_elapsed = (
                elapsed_secs_for_up_front_computations
                + elapsed_secs_for_cavi_iterations
            )
            for (l, link) in enumerate(links_for_category_probabilities):
                update_performance_results(
                    performance_over_time_as_dict,
                    covariates_train=design_matrix_train,
                    labels_train=labels,
                    beta_mean=variational_params.beta.mean,
                    secs_elapsed=secs_elapsed,
                    link=link,
                    covariates_test=design_matrix_test,
                    labels_test=labels_test,
                    update_secs_elapsed=not l,
                )

            end_time_for_computing_these_cat_probs = time.time()
            elapsed_secs_for_computing_category_probabilities += (
                end_time_for_computing_these_cat_probs
                - start_time_for_computing_these_cat_probs
            )
        performance_over_time_as_dict["ELBO (mean over N,K)"].append(
            elbo_stats.mean_elbo
        )
        performance_over_time_as_dict["seconds elapsed (category probs)"].append(
            elapsed_secs_for_computing_category_probabilities
        )

        if verbose:
            info_to_display = (
                f"Iteration: {n_iterations_so_far}. \n"
                f"{get_most_recent_performance_results_as_string(performance_over_time_as_dict)}"
            )
            print(info_to_display)
            sys.stdout.flush()

    performance_over_time = pd.DataFrame(performance_over_time_as_dict)
    return CAVI_Results(variational_params, performance_over_time)
