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
from scipy.sparse import isspmatrix, spmatrix

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.ib_cavi.multi.ib_logit.elbo import compute_elbo
from categorical_from_binary.ib_cavi.multi.inference_wrapper import (
    compute_ib_cavi,
    compute_precomputables_dummy,
)
from categorical_from_binary.ib_cavi.multi.structs import (
    CAVI_Results,
    ELBO_Stats,
    Precomputables,
    PriorType,
    VariationalBeta,
    VariationalOmegas,
    VariationalParams,
)
from categorical_from_binary.selection.hyperparameters import Hyperparameters
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def _initialize_beta_covariance_for_one_category(
    design_matrix: NumpyArray2D,
) -> NumpyArray2D:
    """
    Assumes that the prior covariance is the identity matrix
    """
    X = design_matrix
    beta_dim_per_category = np.shape(X)[1]
    data_precision = X.T @ X
    prior_precision = scipy.sparse.identity(beta_dim_per_category)
    sum_of_precisions = data_precision + prior_precision
    return np.asarray(np.linalg.inv(sum_of_precisions))


def update_omegas(
    variational_beta: VariationalBeta,
    design_matrix: NumpyArray2D,
    labels: NumpyArray2D,
) -> VariationalOmegas:
    num_samples, num_categories = np.shape(labels)
    X = design_matrix
    qmu, qSigma = variational_beta.mean, variational_beta.cov
    c = np.zeros((num_samples, num_categories))
    for i in range(num_samples):
        x_i = X[i, :]
        for k in range(num_categories):
            c[i, k] = np.sqrt(x_i.T @ qSigma[:, :, k] @ x_i + (x_i.T @ qmu[:, k]) ** 2)

    b = np.ones_like(c)
    return VariationalOmegas(b, c)


def compute_expected_value_of_polya_gamma_random_variable(b: float, c: float):
    if b <= 0:
        raise ValueError("The b parameter of a PG distribution must be non-negative.")
    return (b / (2 * c)) * (np.exp(c) - 1) / (np.exp(c) + 1)


def compute_expected_omegas(variational_omegas: VariationalOmegas) -> NumpyArray2D:
    num_samples, num_categories = np.shape(variational_omegas.b)
    expected_omegas = np.zeros_like(variational_omegas.b)
    for i in range(num_samples):
        for k in range(num_categories):
            expected_omegas[
                i, k
            ] = compute_expected_value_of_polya_gamma_random_variable(
                variational_omegas.b[i, k], variational_omegas.c[i, k]
            )
    return expected_omegas


def update_beta_covariance_for_one_category(
    design_matrix: NumpyArray2D,
    expected_omegas_for_category: NumpyArray1D,
    prior_precision_for_one_category: Optional[NumpyArray2D] = None,
) -> NumpyArray2D:
    """
    Returns:
        numpy array with shape (M,M), where M is the number of covariates
    """
    X = design_matrix
    num_covariates = np.shape(X)[1]
    if prior_precision_for_one_category is None:
        prior_precision_for_one_category = np.eye(num_covariates)

    W = scipy.sparse.diags(expected_omegas_for_category)
    precision_for_one_category = prior_precision_for_one_category + X.T @ W @ X
    return np.linalg.inv(precision_for_one_category)


def update_beta(
    variational_omegas: VariationalOmegas,
    design_matrix: NumpyArray2D,
    labels: NumpyArray2D,
    prior_mean: Optional[NumpyArray2D] = None,
    prior_precision: Optional[NumpyArray2D] = None,
) -> VariationalBeta:
    """
    Computes variational parameters (variational mean, variational cov) for beta,
    assuming that our prior has zero-mean - more specifically, that each
        beta_k \sim N(0, prior_precision^{-1}).

    Arguments:
        prior_mean:  has shape (M,K), where M is the number of covariates.
            and K is the number of categories.
        prior_precision:  has shape (M,M), where M is the number of covariates.
            Assumed to be the same for each category.
    """
    X = design_matrix
    expected_omegas = compute_expected_omegas(variational_omegas)
    kappas = labels - 0.5
    num_covariates = np.shape(X)[1]
    num_categories = np.shape(expected_omegas)[1]

    if prior_mean is None:
        prior_mean = np.zeros((num_covariates, num_categories))
    if prior_precision is None:
        prior_precision = np.eye(num_covariates)

    beta_cov_new = np.zeros((num_covariates, num_covariates, num_categories))
    for k in range(num_categories):
        expected_omegas_for_one_category = expected_omegas[:, k]
        beta_cov_new[:, :, k] = update_beta_covariance_for_one_category(
            design_matrix,
            expected_omegas_for_one_category,
            prior_precision,
        )

    beta_mean_new = np.einsum("mnk,nk -> mk", beta_cov_new, design_matrix.T @ kappas)
    return VariationalBeta(beta_mean_new, beta_cov_new)


def initialize_variational_params(
    design_matrix: NumpyArray2D,
    n_categories: int,
    variational_params_init: Optional[VariationalParams] = None,
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

        beta_mean_init = np.zeros((beta_dim_per_category, n_categories))
        # For CBC-Logit, unlike CBC-Probit, the beta covariance differs across categories.
        # The heterogeneity is induced by the polya-gamma augmentation variable.
        # Thus, whereas CBC-Probit inference could use a compact representation of the
        # covariances, with dimension (M,M), here we MUST non-compact representaiton,
        # with dimension (M,M,K),
        beta_cov_init = np.array(
            [
                _initialize_beta_covariance_for_one_category(design_matrix)
                for i in range(n_categories)
            ]
        ).T

        variational_beta_init = VariationalBeta(beta_mean_init, beta_cov_init)

    return VariationalParams(variational_beta_init)


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
    variational_omegas = update_omegas(variational_params.beta, design_matrix, labels)
    variational_beta = update_beta(
        variational_omegas,
        design_matrix,
        labels,
        prior_beta_mean,
        prior_beta_precision,
    )
    variational_params = VariationalParams(variational_beta, omegas=variational_omegas)

    ### update elbo
    n_samples, n_categories = np.shape(labels)
    if elbo_stats.convergence_criterion_drop_in_mean_elbo != -np.inf:
        elbo = compute_elbo(
            variational_params.beta.mean,
            variational_params.beta.cov,
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


def compute_multiclass_logit_vi_with_polya_gamma_augmentation(
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
    verbose: bool = True,
) -> CAVI_Results:
    """
    Variational inference for Bayesian CBC-Logit regression with normal prior on regression coefficients.

    Currently assumes prior is beta_k~N(0,I) for each vector of
    regression weights, k=1,..,K, where K is the number of response
    categories.

    The inference is done on an independent binary model (Wojnowicz et al 2021), and so one can choose different category probability
    formulas within which to plug in the resulting betas (CBC Logit and CBM Logit are potentially the ones to study).

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
        prior_beta_mean:  Optional array with shape (M,K), where M is the number of covariates.
            and K is the number of categories.  If None, taken to be the 0 matrix.
        prior_beta_precision:  Optional array with shape (M,M), where M is the number of covariates.
            Assumed to be the same for each category.  If None, taken to be the identity matrix.
        variational_beta_init: Optional[VariationalBeta]
            initial mean is a numpy array with shape (beta_dim, )
            where, if the covariates contain an intercept term,
                beta_dim = n_covariates (in the non-autoregressive case)
                         = (n_covariates - 1) + n_categories (in the AR case, if the covariates contain an intercept term)
                         = n_covariates + n_categories (in the AR case, if the covariates don't contain an intercept term)

    Returns:
        CAVI_Results. Includes VariationalParameters.  Which in turn includes:
            beta_mean, beta_cov: The variational parameters for the normal distribution.
                beta_mean is an array with shape (beta_dim, n_categories)
                beta_cov is an array with shape (beta_dim, beta_dim)
                    note that the covariance is the same for each category, so
                    we just return a single beta_cov instead of an array
                    of them with shape (n_categories, )
    """

    # convert to dense if sparse
    if isspmatrix(covariates):
        warnings.warn("Converting covariates to dense")
        covariates = np.array(covariates.todense())
    if isspmatrix(labels):
        warnings.warn("Converting labels to dense")
        labels = np.array(labels.todense(), dtype=int)
    if isspmatrix(covariates_test):
        covariates_test = np.array(covariates_test.todense())
    if isspmatrix(labels_test):
        labels_test = np.array(labels_test.todense(), dtype=int)

    links_for_category_probabilities = [Link.CBC_LOGIT, Link.CBM_LOGIT]

    return compute_ib_cavi(
        labels,
        covariates,
        links_for_category_probabilities,
        initialize_variational_params,
        update_variational_params_and_elbo_stats,
        compute_precomputables_dummy,
        max_n_iterations,
        convergence_criterion_drop_in_mean_elbo,
        use_autoregressive_design_matrix,
        labels_test,
        covariates_test,
        prior_beta_mean,
        prior_beta_precision,
        variational_params_init,
        verbose,
    )
