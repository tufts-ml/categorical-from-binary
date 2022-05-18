"""
Here we do coordinate ascent variational inference (CAVI)
on the IB model (Wojnowicz et al 2021); the resulting distribution on 
betas can be used within the CBC-Probit model (Johndrow et al 2013)
or the CBM-Probit model (Wojnowicz et al 2021)
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import scipy
from scipy.sparse import spmatrix
from scipy.special import log_ndtr as log_norm_cdf_fast
from scipy.stats import norm

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.ib_cavi.multi.ib_probit.elbo import compute_elbo
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.shrinkage_groups import (
    ShrinkageGroupingStrategy,
    make_shrinkage_groups,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.tau_helpers import (
    compute_expected_tau_reciprocal_array,
)
from categorical_from_binary.ib_cavi.multi.inference_wrapper import compute_ib_cavi
from categorical_from_binary.ib_cavi.multi.structs import (
    CAVI_Results,
    ELBO_Stats,
    Precomputables,
    PriorType,
    VariationalBeta,
    VariationalParams,
    VariationalTaus,
)
from categorical_from_binary.ib_cavi.trunc_norm import (
    compute_expected_value_normal_minus,
    compute_expected_value_normal_plus,
)
from categorical_from_binary.math import logdiffexp
from categorical_from_binary.selection.hyperparameters import Hyperparameters
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


def compute_variational_expectation_of_z(
    labels: Union[NumpyArray2D, spmatrix],
    design_matrix: Union[NumpyArray2D, spmatrix],
    beta_mean: Union[NumpyArray2D, spmatrix],
) -> Union[NumpyArray2D, spmatrix]:
    """
    Wrapper function that directs flow to either the sparse or dense variant,
    depending on the nature of the inputs.

    To get better intuition on the underlying computation,
    see `compute_variational_expectation_of_z_in_an_intuitive_way`.
    """
    if scipy.sparse.issparse(labels) or scipy.sparse.issparse(design_matrix):
        return compute_variational_expectation_of_z_with_sparse_inputs(
            labels, design_matrix, beta_mean
        )
    else:
        return compute_variational_expectation_of_z_with_dense_inputs(
            labels, design_matrix, beta_mean
        )


def compute_variational_expectation_of_z_with_dense_inputs(
    labels: NumpyArray2D,
    design_matrix: NumpyArray2D,
    beta_mean: NumpyArray2D,
) -> NumpyArray2D:

    eta = design_matrix @ beta_mean
    # A more direct, but less numerically stable, way to code up what's happening here:
    # pdfs = norm.pdf(-eta)
    # cdfs = norm_cdf_fast(-eta)
    # expected_zs_if_all_categories_were_observed = eta + pdfs / (1 - cdfs)
    # expected_zs_if_no_categories_were_observed = eta - pdfs / cdfs
    log_pdfs = norm.logpdf(-eta)
    log_cdfs = log_norm_cdf_fast(-eta)
    log_sfs = logdiffexp(0, log_cdfs)  # sf = 1 -cdf
    expected_zs_if_all_categories_were_observed = eta + np.exp(log_pdfs - log_sfs)
    expected_zs_if_no_categories_were_observed = eta - np.exp(log_pdfs - log_cdfs)
    return (
        labels * expected_zs_if_all_categories_were_observed
        + (1 - labels) * expected_zs_if_no_categories_were_observed
    )


def compute_variational_expectation_of_z_with_sparse_inputs(
    labels: Union[NumpyArray2D, spmatrix],
    design_matrix: Union[NumpyArray2D, spmatrix],
    beta_mean: Union[NumpyArray2D, spmatrix],
) -> spmatrix:
    """
    Designed for use when at least one of the inputs is sparse.  Since the others may not be
    sparse, we force everything to be sparse before proceeding.
    """
    n_samples, n_categories = np.shape(labels)

    labels = scipy.sparse.csr_matrix(labels)
    eta = scipy.sparse.csr_matrix(design_matrix) @ scipy.sparse.csr_matrix(beta_mean)

    # See report on categorical modeling for definition of EZ_star and EZ_daggers; basically we decompose EZ
    # into two additive factors correpsnding to components where eta_ik =0 and eta_ik !=0.

    # First we compute `EZ_daggers`.  EZ_daggers gives the part of EZ corresponding to entries where eta==0
    EZ_daggers = 2 * norm.pdf(0) * (eta == 0).multiply(labels == 1) - 2 * norm.pdf(
        0
    ) * (eta == 0).multiply(labels == 0)
    if eta.nnz == 0:
        return EZ_daggers

    # Now we compute `EZ_star`.  EZ_star gives the part of EZ corresponding to entries where eta!=0
    log_pdfs_for_nzvals_of_eta = norm.logpdf(-eta.data)
    log_cdfs_for_nzvals_of_eta = log_norm_cdf_fast(-eta.data)
    log_sfs_for_nzvals_of_eta = logdiffexp(0, log_cdfs_for_nzvals_of_eta)  # sf = 1 -cdf
    expected_zs_for_nzvals_of_eta_if_all_categories_were_observed = eta.data + np.exp(
        log_pdfs_for_nzvals_of_eta - log_sfs_for_nzvals_of_eta
    )
    expected_zs_for_nzvals_of_eta_if_no_categories_were_observed = eta.data - np.exp(
        log_pdfs_for_nzvals_of_eta - log_cdfs_for_nzvals_of_eta
    )
    row_idxs_for_nzvals_of_eta, col_idxs_for_nzvals_of_eta = eta.nonzero()
    labels_for_nzvals_of_eta = np.array(
        labels[row_idxs_for_nzvals_of_eta, col_idxs_for_nzvals_of_eta]
    )
    expected_z_for_nzvals_of_eta = (
        labels_for_nzvals_of_eta
        * expected_zs_for_nzvals_of_eta_if_all_categories_were_observed
        + (1 - labels_for_nzvals_of_eta)
        * expected_zs_for_nzvals_of_eta_if_no_categories_were_observed
    ).flatten()
    EZ_star = scipy.sparse.coo_matrix(
        (
            expected_z_for_nzvals_of_eta,
            (row_idxs_for_nzvals_of_eta, col_idxs_for_nzvals_of_eta),
        ),
        shape=(n_samples, n_categories),
    )

    EZ_sparse = EZ_star + EZ_daggers
    return EZ_sparse


def compute_variational_expectation_of_z_in_an_intuitive_way(
    labels: NumpyArray2D,
    design_matrix: NumpyArray2D,
    beta_mean: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        labels: array with shape (n_obs, n_categories)
            one-hot encoded representations of response categories
        beta_mean: has shape (M,K), where M is the number of covariates and K is
            the number of categories.
    """
    n_samples, n_categories = np.shape(labels)
    choices = np.argmax(labels, 1)  # index of selected category
    z_expected = np.zeros((n_samples, n_categories))
    eta = design_matrix @ beta_mean
    # This is unnecessarily slow for large N.  See `compute_variational_expectation_of_z`
    # for a vectorized approach.
    for i in range(n_samples):
        for k in range(n_categories):
            if choices[i] == k:
                z_expected[i, k] = compute_expected_value_normal_plus(eta[i, k])
            else:
                z_expected[i, k] = compute_expected_value_normal_minus(eta[i, k])
    return z_expected


def update_taus(
    variational_beta: VariationalBeta,
    hyperparameters: Hyperparameters,
    shrinkage_grouping_matrix: NumpyArray2D,
) -> VariationalTaus:
    num_covariates, num_categories = np.shape(variational_beta.mean)
    hp = hyperparameters
    qbeta_mean = variational_beta.mean  # MxK
    qbeta_vars = np.array(
        [np.diag(variational_beta.cov[:, :, k]) for k in range(num_categories)]
    ).T  # MxK

    a = np.zeros((num_covariates, num_categories))
    d = np.zeros((num_covariates, num_categories))
    for m in range(num_covariates):
        for k in range(num_categories):
            shrinkage_group = shrinkage_grouping_matrix[m, k]
            size_of_shrinkage_group = np.sum(
                shrinkage_grouping_matrix == shrinkage_group
            )
            a[m, k] = hp.lambda_ - 0.5 * size_of_shrinkage_group

            shrinkage_group_member_indices = np.where(
                shrinkage_grouping_matrix == shrinkage_group
            )
            d[m, k] = np.sum(
                qbeta_mean[shrinkage_group_member_indices] ** 2
                + qbeta_vars[shrinkage_group_member_indices]
            )

    return VariationalTaus(
        a=a,
        c=1 / (hp.gamma**2),
        d=d,
    )


def _compute_covariance_of_beta_under_normal_prior(
    design_matrix: NumpyArray2D,
):
    """
    Returns:
        A sparse matrix if `design_matrix` was apsrse, and a dense matrix if `design_matrix` was dense.
    """
    return_sparse_matrix = False
    if scipy.sparse.issparse(design_matrix):
        return_sparse_matrix = True

    X = design_matrix
    beta_dim_per_category = np.shape(X)[1]
    data_precision = X.T @ X
    prior_precision = scipy.sparse.identity(beta_dim_per_category)
    sum_of_precisions = data_precision + prior_precision
    # Warning: Sparse operations return an np.matrix, which a special case of np.array which
    # has different overwritten interpretations of basic operations like `*`. For more
    # information, see `https://towardsdatascience.com/6-key-differences-between-np-ndarray-and-np-matrix-objects-e3f5234ae327`
    # Ugh.  OOP for the LOSS.  FTL.
    # Note that the np.matrix type can be fixed via np.asarray().
    if return_sparse_matrix:
        return scipy.sparse.linalg.inv(sum_of_precisions)  # .toarray()
    else:
        return np.asarray(np.linalg.inv(sum_of_precisions))


def compute_variational_parameters_for_beta_under_normal_prior(
    design_matrix: NumpyArray2D,
    z_expected: NumpyArray2D,
    beta_cov_init: Union[NumpyArray2D, NumpyArray3D],
    beta_cov_init_times_design_matrix_transposed: Optional[
        Union[NumpyArray2D, spmatrix]
    ] = None,
) -> VariationalBeta:
    """
    Computes variational parameters (variational mean, variational cov) for beta,
    assuming that our prior is normal - more specifically, N(0,I).

    Notes:
        * The covariance for beta is not updated here - it is fully determined by the prior
        and the design matrix.
        * If beta_cov_init is NumpyArray2D, then we are using a "compact" representation for covariances.
        The compact representation can be used to save space when the covariance matrix is identical
        across categories (as it is when doing variational inference on a  standard *CBC-Probit model)

    Arguments:
        beta_cov_init_times_design_matrix_transposed: Optional, can be provided to save on computation time,
            since these matrices may be large.
    """
    covariance_is_compact = np.ndim(beta_cov_init) == 2
    if covariance_is_compact:
        if beta_cov_init_times_design_matrix_transposed is not None:
            beta_mean_new = beta_cov_init_times_design_matrix_transposed @ z_expected
        else:
            beta_mean_new = beta_cov_init @ design_matrix.T @ z_expected
    else:
        beta_mean_new = np.einsum(
            "mnk,nk -> mk", beta_cov_init, design_matrix.T @ z_expected
        )
    return VariationalBeta(beta_mean_new, beta_cov_init)


def compute_variational_parameters_for_beta_under_normal_gamma_prior(
    X: NumpyArray2D,
    z_mean: NumpyArray2D,
    variational_taus: VariationalTaus,
    XtX: Optional[NumpyArray2D] = None,
) -> VariationalBeta:
    """
    Note:
        Because we are sharing sparsity information (about covariates) across categories,
        there is a single covariance matrix for all categories.
    """
    expected_tau_reciprocal_array = compute_expected_tau_reciprocal_array(
        variational_taus,
    )

    M, K = np.shape(expected_tau_reciprocal_array)
    if XtX is None:
        XtX = X.T @ X
    beta_cov = np.zeros((M, M, K))
    for k in range(K):
        beta_cov[:, :, k] = np.linalg.inv(expected_tau_reciprocal_array[:, k] + XtX)
    # The last entry in `einsum` assumes the prior mean is 0.
    beta_mean = np.einsum("mnk,nk -> mk", beta_cov, X.T @ z_mean)
    return VariationalBeta(beta_mean, beta_cov)


def initialize_variational_params(
    design_matrix: NumpyArray2D,
    n_categories: int,
    variational_params_init: Optional[VariationalParams] = None,
    prior_type: PriorType = PriorType.NORMAL,
) -> VariationalParams:

    beta_dim_per_category = np.shape(design_matrix)[1]

    if variational_params_init is not None and variational_params_init.beta is not None:
        variational_beta_init = variational_params_init.beta
    else:
        warnings.warn(
            f"Initial value for variational expectation of beta is the matrix of all 0's, "
            f"and for the covariance matrix is the identity matrix"
        )
        # TODO: Add support for warm-starting the inference with the MLE
        # variational initialization
        variational_beta_init = VariationalBeta(
            mean=np.zeros((beta_dim_per_category, n_categories)),
            cov=np.eye(beta_dim_per_category),
        )
        # Note: the above assumes that we want a "compact" representation of the covariance -- i.e.
        # that it has dimensionality (M,M). If we're using a non-compact representaiton, with dimension (M,M,K),
        # then we want to use
        #
        # variational_beta_init = VariationalBeta(
        #     mean=np.zeros((beta_dim_per_category, n_categories)),
        #     cov=np.array(
        #         [np.eye(beta_dim_per_category) for i in range(n_categories)]
        #     ).T,
        # )

    if prior_type == PriorType.NORMAL:
        # We start with the optimal beta_cov, because this does not change over inference.
        # Note that it is identical across categories.
        #
        # Note that the first term in the sum below assumes the prior covariance matrix is I.

        # The matrix`variational_beta_init.cov`, and therefore also `beta_cov_init_times_design_matrix_transposed `,
        # will be sparse if `design_matrix` is sparse.
        variational_beta_init.cov = _compute_covariance_of_beta_under_normal_prior(
            design_matrix
        )
    return VariationalParams(variational_beta_init)


def compute_precomputables(
    variational_params_init: VariationalParams,
    design_matrix: NumpyArray2D,
    prior_type: PriorType = PriorType.NORMAL,
    shrinkage_grouping_strategy: Optional[ShrinkageGroupingStrategy] = None,
) -> Precomputables:

    beta_cov_init_times_design_matrix_transposed = (
        variational_params_init.beta.cov @ design_matrix.T
    )

    if prior_type == PriorType.NORMAL_GAMMA:
        # initialize shrinkage grouping strategy
        shrinkage_grouping_matrix = make_shrinkage_groups(
            shrinkage_grouping_strategy, variational_params_init.beta.mean
        )
        # side thing: compute XtX as it will be used later
        design_matrix_t_design_matrix = design_matrix.T @ design_matrix

    else:
        shrinkage_grouping_matrix = None
        design_matrix_t_design_matrix = None
    return Precomputables(
        beta_cov_init_times_design_matrix_transposed,
        shrinkage_grouping_matrix,
        design_matrix_t_design_matrix,
    )


def update_variational_params_and_elbo_stats(
    variational_params: VariationalParams,
    elbo_stats: ELBO_Stats,
    design_matrix: NumpyArray2D,
    labels: NumpyArray2D,
    prior_beta_mean: NumpyArray2D,
    prior_beta_precision: NumpyArray2D,
    precomputables: Precomputables,
    hyperparameters: Hyperparameters,
    prior_type: PriorType = PriorType.NORMAL,
) -> Tuple[VariationalParams, ELBO_Stats]:
    """
    We violate Uncle Bob's one-action-per-function edict here for efficiency reasons:
    intermediate computations for variational updates can be used for the ELBO computation
    """
    z_expected = compute_variational_expectation_of_z(
        labels,
        design_matrix,
        variational_params.beta.mean,
    )
    if prior_type == PriorType.NORMAL:
        variational_taus = None
        variational_beta = compute_variational_parameters_for_beta_under_normal_prior(
            design_matrix,
            z_expected,
            variational_params.beta.cov,
            precomputables.beta_cov_init_times_design_matrix_transposed,
        )
    elif prior_type == PriorType.NORMAL_GAMMA:
        variational_taus = update_taus(
            variational_params.beta,
            hyperparameters,
            precomputables.shrinkage_grouping_matrix,
        )
        variational_beta = (
            compute_variational_parameters_for_beta_under_normal_gamma_prior(
                design_matrix,
                z_expected,
                variational_taus,
                precomputables.design_matrix_t_design_matrix,
            )
        )
    else:
        raise ValueError(f"Prior type {prior_type} is unknown.")
    variational_params = VariationalParams(variational_beta, taus=variational_taus)

    ### update elbo
    n_samples, n_categories = np.shape(labels)
    if elbo_stats.convergence_criterion_drop_in_mean_elbo != -np.inf:
        elbo = compute_elbo(
            variational_params.beta.mean,
            variational_params.beta.cov,
            z_expected,
            design_matrix,
            labels,
        )
        mean_elbo = elbo / n_samples * n_categories
        drop_in_mean_elbo = mean_elbo - elbo_stats.previous_mean_elbo
        previous_mean_elbo = mean_elbo
        elbo_stats = ELBO_Stats(
            elbo_stats.convergence_criterion_drop_in_mean_elbo,
            previous_mean_elbo,
            mean_elbo,
            drop_in_mean_elbo,
        )

    return variational_params, elbo_stats


def compute_multiclass_probit_vi_with_normal_gamma_prior(
    labels: NumpyArray2D,
    covariates: NumpyArray2D,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_mean_elbo: float = -np.inf,
    use_autoregressive_design_matrix: bool = False,
    labels_test: Optional[NumpyArray2D] = None,
    covariates_test: Optional[NumpyArray2D] = None,
    prior_beta_mean: Optional[NumpyArray2D] = None,
    prior_beta_precision: Optional[NumpyArray2D] = None,
    variational_params_init: Optional[VariationalParams] = None,
    hyperparameters: Optional[Hyperparameters] = None,
    shrinkage_grouping_strategy: Optional[
        ShrinkageGroupingStrategy
    ] = ShrinkageGroupingStrategy.FREE,
    verbose: bool = True,
) -> CAVI_Results:
    """
    Variational inference for Bayesian multiclass probit regression with normal-gamma prior
    on regression coefficients.

    Notes:
        - Special handling of sparse matrices not yet supported for this choice of prior.
    """
    links_for_category_probabilities = [Link.CBM_PROBIT, Link.CBC_PROBIT]
    prior_type = PriorType.NORMAL_GAMMA
    return compute_ib_cavi(
        labels,
        covariates,
        links_for_category_probabilities,
        initialize_variational_params,
        update_variational_params_and_elbo_stats,
        compute_precomputables,
        max_n_iterations,
        convergence_criterion_drop_in_mean_elbo,
        use_autoregressive_design_matrix,
        labels_test,
        covariates_test,
        prior_beta_mean,
        prior_beta_precision,
        variational_params_init,
        prior_type,
        hyperparameters,
        shrinkage_grouping_strategy,
        verbose,
    )


def compute_multiclass_probit_vi_with_normal_prior(
    labels: Union[NumpyArray2D, spmatrix],
    covariates: Union[NumpyArray2D, spmatrix],
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_mean_elbo: float = -np.inf,
    use_autoregressive_design_matrix: bool = False,
    labels_test: Optional[NumpyArray2D] = None,
    covariates_test: Optional[NumpyArray2D] = None,
    prior_beta_mean: Optional[NumpyArray2D] = None,
    prior_beta_precision: Optional[NumpyArray2D] = None,
    variational_params_init: Optional[VariationalParams] = None,
    save_beta_every_secs: Optional[float] = None,
    save_beta_dir: Optional[str] = None,
    verbose: bool = True,
) -> CAVI_Results:
    """
    Variational inference for Bayesian multiclass probit regression with normal prior on regression coefficients.

    Currently assumes prior is beta_k~N(0,I) for each vector of
    regression weights, k=1,..,K, where K is the number of response
    categories.

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
    links_for_category_probabilities = [Link.CBM_PROBIT, Link.CBC_PROBIT]
    prior_type = PriorType.NORMAL
    return compute_ib_cavi(
        labels,
        covariates,
        links_for_category_probabilities,
        initialize_variational_params,
        update_variational_params_and_elbo_stats,
        compute_precomputables,
        max_n_iterations,
        convergence_criterion_drop_in_mean_elbo,
        use_autoregressive_design_matrix,
        labels_test,
        covariates_test,
        prior_beta_mean,
        prior_beta_precision,
        variational_params_init,
        prior_type,
        save_beta_every_secs=save_beta_every_secs,
        save_beta_dir=save_beta_dir,
        verbose=verbose,
    )
