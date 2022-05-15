"""
Generate some multiclass regression data
"""
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy

from categorical_from_binary.data_generation.bayes_binary_reg import (
    LinearPredictorScaleAllocator,
    generate_features_via_independent_normals,
    generate_regression_coefficients_for_features as generate_regression_coefficients_for_features_for_one_column,
)
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.kl import sigmoid
from categorical_from_binary.numpy_helpers import enforce_bounds_on_prob_vector
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


# TODO: Move 'types' to more central place than simple_hdpmm

# TODO: The logistic regression modules defines 'features' without the all ones column;
# This module defines 'features' as having the ones column.  Make it consistent across modules.


class Link(int, Enum):
    """
    The CBM models are Categorical From Binary via Marginalization models (see arxiv paper)
    The CBC models Categorical From Binary via Conditioning models (see arxiv paper)
    Multi Probit uses the technique on pp.120 of Kenneth Train's book Discrete Choice Models with Simulations

    NOTE:
        Inherting from `int` as well as `Enum` makes this json serializable
    """

    MULTI_LOGIT = 1
    STICK_BREAKING = 2
    CBC_PROBIT = 3
    CBM_PROBIT = 4
    MULTI_PROBIT = 5
    CBC_LOGIT = 6
    CBM_LOGIT = 7
    SOFTMAX = 8
    BMA_PROBIT = 9
    BMA_LOGIT = 10


@dataclass
class ControlCategoryPredictability:
    """
    Let's suppose we map each covariate to a category.
        So high values of covariate 1 or 2 might "signal" category 1, etc.
        So high values of covariate 3 or 4 might "signal" category 2, etc.
    Then, generate covariates by:
        1. Selecting a subset of covariates to "matter"
        2. Draw these from a Normal(0, scale_for_predictive_categories)
        3. Draw remaining covariates from a Normal(0, scale_for_non_predictive_categories).
    """

    scale_for_predictive_categories: Optional[float] = 1.0
    scale_for_non_predictive_categories: Optional[float] = 0.001


@dataclass
class MakeUniformCategories:
    """
    Should be considered deprecated.  A better option would be to use ControlCategoryPredictability,
    with equal values for `scale_for_predictive_categories` and `scale_for_nonpredictive_categories`.
    """

    mean: Optional[float] = 0.0
    scale_for_linear_predictor: Optional[float] = 1.0
    linear_predictor_scale_allocator: Optional[
        LinearPredictorScaleAllocator
    ] = LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES


@dataclass
class MakePreferredCategories:
    """
    Should be considered deprecated. A more successful attempt at reaching the same goal was realized by
    `ControlCategoryPredictability`.

    Original idea: When generating entries for a beta matrix with shape (n_features+1, num_beta_columns),
    where num_beta_columns depends on the number of categories in a way determined by the link function,
    we make the scale of the sampling dependent on the column.  This leads some columns to be preferred over others.
    However, the value of features can still play a role, which allows for there to be variance across samples.
    """

    mean: Optional[float] = 0.0
    scale_for_linear_predictor: Optional[float] = 1.0
    linear_predictor_scale_allocator: Optional[
        LinearPredictorScaleAllocator
    ] = LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES


BetaCategoryStrategy = Union[
    MakeUniformCategories, MakePreferredCategories, ControlCategoryPredictability
]
"""
`BetaCategoryStrategy` enumerates ways so set variances on the beta coefficients in an attempt to try to make some categories
used more than others.
"""


def get_num_beta_columns(link: Link, n_categories: int) -> int:
    """
    The beta matrix has shape (n_features+1, num_beta_columns),
    The num_beta_columns depends on the number of categories in a way determined by the link function.
    The value differs due to identifiability considerations.

    This function determines the number of beta columns based on the link function and
    the number of response categories.
    """
    if link == Link.MULTI_LOGIT or link == Link.STICK_BREAKING:
        num_beta_columns = n_categories - 1
    elif (
        link == Link.CBC_PROBIT
        or link == Link.CBM_PROBIT
        or link == Link.MULTI_PROBIT
        or link == Link.CBC_LOGIT
        or link == Link.CBM_LOGIT
        or link == Link.SOFTMAX
    ):
        num_beta_columns = n_categories
    else:
        raise NotImplementedError
    return num_beta_columns


@dataclass
class MulticlassRegressionDataset:
    """
    Aka Categorical or Polytomous Regression Dataset

    Attributes:

        features: np.ndarray  # shape:  (n_samples, n_features_non_autoregressive)
            Includes ones column added for intercept term!
            Does NOT include autoregressive features
            See data_generation.data
        labels: np.ndarray  # shape: (n_samples, n_categories)
            Each row has exactly one 1.
        beta: np.ndarray  # shape: (n_features_non_autoregressive, num_beta_columns)
            Regression weights.
            The 0th row is the intercept term, if it exists
            num_beta_columns depends on the link function:
                num_beta_columns = n_categories - 1 for Link.MULTI_LOGIT and Link.STICK_BREAKING
                num_beta_columns = n_categories for Link.CBC_PROBIT or Link.CBM_PROBIT or Link.CBC_LOGIT or Link.CBM_LOGIT
            This is optional because for a real dataset it won't be known.
        link: Link
            The link function used to generate the responses from the features
            and beta. Important for interpreting the beta coefficients.
            This is optional because for a real dataset it won't be known.
        seed: int
            The numpy seed used for the function
            This is optional because for a real dataset it won't be known.
    """

    features: NumpyArray2D
    labels: NumpyArray2D
    beta: Optional[NumpyArray2D] = None
    link: Optional[Link] = None
    seed: Optional[int] = None


def generate_multiclass_regression_dataset(
    n_samples: int,
    n_features: int,
    n_categories: int,
    beta_0: Optional[NumpyArray1D] = None,
    link: Link = Link.MULTI_LOGIT,
    n_sparse_features: int = 0,
    n_categories_where_all_beta_coefficients_are_sparse: Optional[int] = None,
    n_bounded_from_zero_features: int = 0,
    lower_bound_for_bounded_from_zero_features: Optional[float] = None,
    beta_category_strategy: Optional[
        BetaCategoryStrategy
    ] = ControlCategoryPredictability(),
    mean_for_intercept: float = 0.0,
    scale_for_intercept: float = 0.25,
    seed: int = None,
    include_intercept: bool = True,
) -> MulticlassRegressionDataset:
    """
    Arguments:
        n_features:
            This is the number of features or predictors, not including the intercept.
        beta_0:
            Optional.  If present, we don't simulate beta_0 randomly, but use the provided value instead.
        link:
            The link function used for the multiclass logistic regression.  Note that the link function
            determines the dimensionality of beta
        include_intercept:
            If True, the 0-th row of the beta matrix will correspond to the intercept, and the 0-th column
            of the features matrix will be a column of all 1's.  If False, neither condition will be true.
    """
    beta = generate_regression_coefficients(
        n_features,
        n_categories,
        link,
        beta_0,
        n_sparse_features,
        n_categories_where_all_beta_coefficients_are_sparse,
        n_bounded_from_zero_features,
        lower_bound_for_bounded_from_zero_features,
        beta_category_strategy,
        mean_for_intercept,
        scale_for_intercept,
        seed,
        include_intercept,
    )

    # generate features and labels
    features_without_ones_column = generate_features_via_independent_normals(
        n_samples, n_features, mean=0.0, scale=1.0
    )

    if include_intercept:
        features = prepend_features_with_column_of_all_ones_for_intercept(
            features_without_ones_column
        )
    else:
        features = features_without_ones_column

    labels = generate_multiclass_labels_for_nonautoregressive_case(
        features,
        beta,
        link,
    )
    return MulticlassRegressionDataset(features, labels, beta, link, seed)


def make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric(
    n_categories: int,
):
    """
    When our multiclass logistic regression uses the stick-breaking link instead of the
    canonical multiclass-logit (aka softmax) link, we can no longer make category probablilities
    symmetric (in the no-feature case) by setting beta_0's equal to 0.

    Here we provide the beta_0's to make the category probabilities symmetric in the no-feature case.

    For the full-feature case, if
        * E[beta_k^0] = b_k, where b_k is what this function returns,
        * E[beta_k^m] = 0 for categories k=1,...,K-1 and covariates m=1,..,M-1
    then
        * E[nu_{ik}] = b_k for all observations i=1,..,N.

    But note that due to the sigmoid nonlinearity I don't think we have symmetric category probabilities anymore.
    """
    K = n_categories
    return np.array([-np.log(K - k) for k in range(1, K)])


def generate_intercepts(
    link: Link,
    n_categories: int,
    num_beta_columns: int,
    n_columns_with_nonzero_beta_coeffs: int,
    mean_for_intercept: float,
    scale_for_intercept: float,
    beta_category_strategy: BetaCategoryStrategy,
) -> NumpyArray1D:
    """
    Arguments:
        num_beta_columns: The number of columns for the beta matrix (covariates x columns).
            The number of columns is affected by:
                * link function -- e.g. multi-logit has K-1 columns whereas do-probit has K columns,
                    where K is the number of categories.
         n_columns_with_nonzero_beta_coeffs: This accounts for
            * user-desired sparsity - callers can force a certain number of categories to have 0 values
                for all regression coefficients (including intercepts)
    """
    # regardless of link, we want the prior expectation of beta_0 to assume symmetry across categories
    if (
        link == Link.MULTI_LOGIT
        or link == Link.CBC_PROBIT
        or link == Link.CBM_PROBIT
        or link == Link.MULTI_PROBIT
        or link == Link.CBC_LOGIT
        or link == Link.CBM_LOGIT
        or link == Link.SOFTMAX
    ):
        # TODO: Make this not hardcoded....Probably should create a NamedTuple where one specifies
        # The link function and the prior mean nad variance on the beta's before sampling.
        beta_oh_means = np.zeros(num_beta_columns) + mean_for_intercept
        beta_oh_sds = np.ones(num_beta_columns) * scale_for_intercept
        if isinstance(beta_category_strategy, MakePreferredCategories):
            # "Effectively" samples matrix entry beta_{mk} to have variance sqrt(1/(k*M)), where m is indexes
            # the covariate and k indexes the cateogry.

            # This strategy makes var(eta_k) = 1/k, so that we can try to make some categories (the larger ones)
            # preferred.
            #
            # To implement, we notice that since the entires of X and entries of beta are all independent with mean
            # zero, then for each category Var(eta) = Var(x'beta) can be reduced to sum_m var(x_m) var(beta_m).
            #
            # The qualifier "effectively" is used because we only range through k's with non-zero coefficients; some
            # may be excluded via `n_categories_where_all_beta_coefficients_are_sparse`.
            for col in range(n_columns_with_nonzero_beta_coeffs):
                multiplier_on_scale_for_beta_column = np.sqrt(
                    (col + 1) / (n_columns_with_nonzero_beta_coeffs)
                )
                beta_oh_sds[col] *= multiplier_on_scale_for_beta_column
        elif isinstance(beta_category_strategy, MakeUniformCategories):
            pass
        elif isinstance(beta_category_strategy, ControlCategoryPredictability):
            pass
        else:
            raise ValueError("I am not sure how to adjust the intercept term by column")

    elif link == Link.STICK_BREAKING:
        warnings.warn(
            f"Ignoring specifed mean for intercept and scale for intercept, because I am calling a function "
            f" that is attempting, albiet poorly, to make category probabilities symmetric.  Also overriding "
            f" the information about desired sparsity contained in `n_columns_with_nonzero_beta_coeffs` "
        )
        beta_oh_means = make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric(
            n_categories,
        )
        # to do: how to determine good variance for this?
        beta_oh_sds = np.ones(n_categories - 1)

        ## Note: I tried using Linderman's compute_psi_cmoments function, but this didn't work
        # as expected...even in the intercepts-only case, it does not produce uniform
        # category probabilities.  Possibly, there is user error
        #
        # from pypolyagamma.utils import compute_psi_cmoments
        # beta_oh_means, beta_oh_vars = compute_psi_cmoments(np.ones(n_categories))
        # beta_oh_sds = np.sqrt(beta_oh_vars)
    else:
        raise ValueError(f"Link is {link}, but I don't know what that is")

    beta_0 = np.zeros(
        num_beta_columns,
    )
    for k in range(n_columns_with_nonzero_beta_coeffs):
        beta_0[k] = np.random.normal(loc=beta_oh_means[k], scale=beta_oh_sds[k], size=1)
    return beta_0


def get_scale_for_beta_column(
    beta_category_strategy: BetaCategoryStrategy,
    n_columns_with_nonzero_beta_coeffs: int,
    column_index: int,
) -> float:
    if isinstance(beta_category_strategy, MakeUniformCategories):
        scale_for_beta_column = beta_category_strategy.scale_for_linear_predictor
    elif isinstance(beta_category_strategy, MakePreferredCategories):
        # This strategy makes var(eta_k) = 1/k * variance_for_linear_predictor,
        # so that we can try to make some categories (the larger ones) preferred.
        #
        # The qualifier "effectively" is used because we only range through k's with non-zero coefficients; some
        # may be excluded via `n_categories_where_all_beta_coefficients_are_sparse`.
        multiplier_on_scale_for_beta_column = np.sqrt(
            (column_index + 1) / (n_columns_with_nonzero_beta_coeffs)
        )
        scale_for_beta_column = (
            beta_category_strategy.scale_for_linear_predictor
            * multiplier_on_scale_for_beta_column
        )
    elif isinstance(beta_category_strategy, ControlCategoryPredictability):
        return np.nan
    else:
        raise ValueError(
            "I cannot determine the scale for sampling regression coefficients for features."
        )
    return scale_for_beta_column


def generate_betas_for_features(
    n_features: int,
    num_beta_columns: int,
    n_columns_with_nonzero_beta_coeffs: int,
    n_sparse_features: int,
    n_bounded_from_zero_features: int,
    lower_bound_for_bounded_from_zero_features: int,
    beta_category_strategy: BetaCategoryStrategy,
    seed: int,
) -> NumpyArray2D:

    if isinstance(beta_category_strategy, MakeUniformCategories) or isinstance(
        beta_category_strategy, MakePreferredCategories
    ):
        betas_for_features = np.zeros((n_features, num_beta_columns))
        for col in range(n_columns_with_nonzero_beta_coeffs):
            scale_for_beta_column = get_scale_for_beta_column(
                beta_category_strategy,
                n_columns_with_nonzero_beta_coeffs,
                column_index=col,
            )
            betas_for_features[
                :, col
            ] = generate_regression_coefficients_for_features_for_one_column(
                n_features,
                n_sparse_features,
                n_bounded_from_zero_features,
                lower_bound_for_bounded_from_zero_features,
                beta_category_strategy.mean,
                scale_for_linear_predictor=scale_for_beta_column,
                linear_predictor_scale_allocator=beta_category_strategy.linear_predictor_scale_allocator,
                seed=seed,
            )
        return betas_for_features
    elif isinstance(beta_category_strategy, ControlCategoryPredictability):
        return generate_betas_for_features_under_predictive_category_strategy(
            n_features,
            num_beta_columns,
            beta_category_strategy.scale_for_predictive_categories,
            beta_category_strategy.scale_for_non_predictive_categories,
        )
    else:
        raise ValueError(
            f"I don't understand the beta category strategy, which is needed to "
            f"generate regresssion coefficients for the features"
        )


def generate_betas_for_features_under_predictive_category_strategy(
    num_features: int,
    num_beta_columns: int,
    scale_for_predictive_categories: float,
    scale_for_non_predictive_categories: float,
) -> NumpyArray2D:
    """
    Idea:
        Let's suppose we map each covariate to a category.
            So high values of covariate 1 or 2 might "signal" category 1, etc.
            So high values of covariate 3 or 4 might "signal" category 2, etc.

        Then, generate covariates by:

        1. Selecting a subset of covariates to "matter"
        2. Draw these from a Normal(0, scale_for_predictive_categories)
        3. Draw remaining covariates from a Normal(0, scale_for_non_predictive_categories)
    """
    if scale_for_non_predictive_categories > scale_for_predictive_categories:
        raise ValueError(
            f"Are you doing this right? scale for predictive categories should "
            f"not be less than the scale for non-predictive categories"
        )

    betas_for_features = np.zeros((num_features, num_beta_columns))
    num_predictive_features_per_category = num_features / num_beta_columns

    for k in range(num_beta_columns):
        for m in range(num_features):
            category_this_feature_predicts = int(
                np.floor(m / num_predictive_features_per_category)
            )

            if k == category_this_feature_predicts:
                scale = scale_for_predictive_categories
            else:
                scale = scale_for_non_predictive_categories
            betas_for_features[m, k] = np.random.normal(loc=0, scale=scale, size=1)
    return betas_for_features


def generate_regression_coefficients(
    n_features: int,
    n_categories: int,
    link: Link,
    beta_0: Optional[NumpyArray1D],
    n_sparse_features: int = 0,
    n_categories_where_all_beta_coefficients_are_sparse: Optional[int] = None,
    n_bounded_from_zero_features: int = 0,
    lower_bound_for_bounded_from_zero_features: Optional[float] = None,
    beta_category_strategy: BetaCategoryStrategy = ControlCategoryPredictability(),
    mean_for_intercept: float = 0.0,
    scale_for_intercept: float = 0.25,
    seed: int = None,
    include_intercept: bool = True,
):
    # generate regression coefficients
    np.random.seed(seed)
    num_beta_columns = get_num_beta_columns(link, n_categories)
    if (
        not n_categories_where_all_beta_coefficients_are_sparse
    ):  # covers 0 or None cases
        n_columns_with_nonzero_beta_coeffs = num_beta_columns
    else:
        n_columns_with_nonzero_beta_coeffs = (
            n_categories - n_categories_where_all_beta_coefficients_are_sparse
        )
        if n_columns_with_nonzero_beta_coeffs > num_beta_columns:
            raise ValueError(
                "Num columns with nonzero coefficients can't exceed the number of beta columns"
            )

    # generate intercepts:
    if beta_0 is None:
        beta_0 = generate_intercepts(
            link,
            n_categories,
            num_beta_columns,
            n_columns_with_nonzero_beta_coeffs,
            mean_for_intercept,
            scale_for_intercept,
            beta_category_strategy,
        )

    # generate regression coefficients for all columns corresponding to categories
    # that are not sparse.
    betas_for_features = generate_betas_for_features(
        n_features,
        num_beta_columns,
        n_columns_with_nonzero_beta_coeffs,
        n_sparse_features,
        n_bounded_from_zero_features,
        lower_bound_for_bounded_from_zero_features,
        beta_category_strategy,
        seed,
    )

    if include_intercept:
        # concatenate the intercept and feature terms into a single beta object
        beta = np.zeros((n_features + 1, num_beta_columns))
        beta[0, :] = beta_0
        beta[1:, :] = betas_for_features
    else:
        beta = betas_for_features
    return beta


def compute_linear_predictors_preventing_downstream_overflow(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    We want to be sure to prevent downstream overflow, since construction of category
    probabilities can make these exponentially large - consider the multi-logit link,
    which constructs category probabilities by exponentiating the linear predictors
    """
    MAX_LINEAR_PREDICTOR = 500  # to prevent overflow; value chosen somewhat arbitrarily, but I know 828 is too large
    eta = features @ beta
    eta[eta > MAX_LINEAR_PREDICTOR] = MAX_LINEAR_PREDICTOR
    return eta


def construct_multi_logit_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories-1)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    n_samples = np.shape(features)[0]
    n_categories = np.shape(beta)[1] + 1
    probs = np.zeros((n_samples, n_categories))
    eta = compute_linear_predictors_preventing_downstream_overflow(features, beta)
    exponentiated_linear_predictors = np.exp(eta)
    normalizers_by_observations = np.sum(exponentiated_linear_predictors, 1) + 1.0
    for k in range(n_categories - 1):
        probs[:, k] = (
            exponentiated_linear_predictors[:, k] / normalizers_by_observations
        )
    probs[:, -1] = 1.0 - np.sum(probs, 1)
    return probs


def construct_non_identified_softmax_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Note: This computation is not the identifiable one used during simulation.

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    M, K = np.shape(beta)
    etas = features @ beta  # N times K

    Z = np.sum(np.array([np.exp(etas[:, k]) for k in range(K)]), 0)
    return np.array([np.exp(etas[:, k]) / Z for k in range(K)]).T


def construct_multi_probit_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
    n_simulations: int = 100000,
    assume_that_random_components_of_utilities_are_independent_across_categories: bool = True,
) -> NumpyArray2D:
    """
    We use a simulated probability method via Lerman and Manski (1981). It is
    summarized very nicely on pp. 120 of Kenneth Train’s book,
    Discrete Choice Models with Simulations:

    We assume the error components are drawn from a multivariate normal N(O,I)

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories-1)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    if not assume_that_random_components_of_utilities_are_independent_across_categories:
        # TODO: Implement this.  Unlike softmax/multi-logit, multi-probit has additional
        # flexibility due to the correlated latent factors, which allows for departure
        # from the IIA assumption.  We might want to investigate behavior in this regime.
        raise NotImplementedError

    n_samples = np.shape(features)[0]
    n_categories = np.shape(beta)[1]
    linear_predictors = features @ beta

    choices_by_simulation = np.zeros((n_samples, n_simulations), dtype=int)
    for s in range(n_simulations):
        # Here is where we assume that the random components are uncorrelated.
        random_contributions = np.random.normal(loc=0, scale=1, size=n_categories)
        random_utilities = (
            linear_predictors + random_contributions
        )  # (n_samples x n_categories)
        choices = np.argmax(random_utilities, 1)  # (n_sample,)
        choices_by_simulation[:, s] = choices

    probs_approx = np.zeros((n_samples, n_categories))
    for k in range(n_categories):
        probs_approx[:, k] = np.mean(choices_by_simulation == k, 1)
    return probs_approx


def construct_stickbreaking_multinomial_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories-1)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    n_samples = np.shape(features)[0]
    n_categories = np.shape(beta)[1] + 1

    # to do, better name
    bernoulli_yes = sigmoid(features @ beta)
    # bernoulli_yes = sigmoid(np.transpose(np.tile(beta_0[:,np.newaxis],n_samples))) #intercept only
    bernoulli_no = 1.0 - bernoulli_yes

    probs = np.ones((n_samples, n_categories))
    for k in range(n_categories - 1):
        for j in range(k):
            probs[:, k] *= bernoulli_no[:, j]
        probs[:, k] *= bernoulli_yes[:, k]

    # the final category is determined by what makes everything sum to 1.0
    probs[:, -1] = 1.0 - np.sum(probs[:, :-1], 1)
    return probs


def construct_cbc_probit_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Notes:
        CBC-probit gives beta an extra element compared to the logit constructions!!!

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    log_cdf = scipy.stats.norm.logcdf
    return _construct_general_cbc_probabilities(features, beta, log_cdf)


def construct_cbc_logit_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Notes:
        CBC-probit gives beta an extra element compared to the logit constructions!!!

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    log_cdf = scipy.stats.logistic.logcdf
    return _construct_general_cbc_probabilities(features, beta, log_cdf)


def _construct_general_cbc_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
    log_cdf: Callable,
) -> NumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)
        log_cdf :  log cumulative distribution function for some symmetric location-scale family.

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    # TODO: align "features" and "covariates"
    #   (sometimes I use the latter to designate the result of prepending the column of all 1's)
    #   (maybe calling this "design_matrix" might be even clearer)
    # TODO: align "utilities" and "linear predictors"

    utilities = compute_linear_predictors_preventing_downstream_overflow(features, beta)
    # TODO: support sparse computation here
    if scipy.sparse.issparse(utilities):
        utilities = utilities.toarray()

    # What this is doing is can perhaps be better understood as follows:
    #
    #  1. Construction of potentials using the CBC-Probit category probs directly from Johndrow et al (2013)
    #  from categorical_from_binary.data_generation.util import  prod_of_columns_except
    #
    #   n_obs = np.shape(features)[0]
    #   n_categories = np.shape(beta)[1]
    #   cdf = scipy.stats.norm.cdf
    #   cdf_neg_utilities = cdf(-utilities)
    #   potentials = np.zeros((n_obs, n_categories))
    #   for k in range(n_categories):
    #         potentials[:, k] = (1 - cdf_neg_utilities[:, k]) * prod_of_columns_except(
    #         cdf_neg_utilities, k
    #       )
    #
    # 2. Alternate construction of potentials, using the expression from the categorical models notes
    #      potentials=cdf(utilities)/cdf(-utilities)
    #
    # The version below though is more numerically stable

    potentials = np.exp(log_cdf(utilities) - log_cdf(-utilities))
    normalizing_constants = np.sum(potentials, 1)
    return potentials / normalizing_constants[:, np.newaxis]


def construct_cbm_probit_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Construct probabiliites using the CBM-Probit model

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    cdf = scipy.stats.norm.cdf
    return _construct_general_cbm_probabilities(features, beta, cdf)


def construct_cbm_logit_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    Construct probabiliites using the CBM-Logit model

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    cdf = scipy.stats.logistic.cdf
    return _construct_general_cbm_probabilities(features, beta, cdf)


def _construct_general_cbm_probabilities(
    features: NumpyArray2D,
    beta: NumpyArray2D,
    cdf: Callable,
) -> NumpyArray2D:
    """
    Construct probabiliites using the CBM model

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)
        cdf : cumulative distribution function for some symmetric location-scale family.

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    # TODO: align "features" and "covariates"
    #   (sometimes I use the latter to designate the result of prepending the column of all 1's)
    #   (maybe calling this "design_matrix" might be even clearer)
    # TODO: align "utilities" and "linear predictors"
    utilities = compute_linear_predictors_preventing_downstream_overflow(features, beta)
    # TODO: support sparse computation here
    if scipy.sparse.issparse(utilities):
        utilities = utilities.toarray()
    cdf_utilities = cdf(utilities)
    return cdf_utilities / np.sum(cdf_utilities, 1)[:, np.newaxis]


# Each value in the `CATEGORY_PROBABILITY_FUNCTION_BY_LINK` dictionary is a function
# whose arguments are features : NumpyArray2D, betas :NumpyArray2D
CATEGORY_PROBABILITY_FUNCTION_BY_LINK = {
    Link.MULTI_LOGIT: construct_multi_logit_probabilities,
    Link.SOFTMAX: construct_non_identified_softmax_probabilities,
    Link.STICK_BREAKING: construct_stickbreaking_multinomial_probabilities,
    Link.CBC_PROBIT: construct_cbc_probit_probabilities,
    Link.CBM_PROBIT: construct_cbm_probit_probabilities,
    Link.MULTI_PROBIT: construct_multi_probit_probabilities,
    Link.CBC_LOGIT: construct_cbc_logit_probabilities,
    Link.CBM_LOGIT: construct_cbm_logit_probabilities,
}


def construct_category_probs(features: NumpyArray2D, beta: NumpyArray2D, link: Link):
    function_to_construct_category_probs = CATEGORY_PROBABILITY_FUNCTION_BY_LINK[link]
    return function_to_construct_category_probs(features, beta)


def compute_mean_log_likelihood(
    features: NumpyArray2D, labels: NumpyArray2D, beta: NumpyArray2D, link: Link
):
    category_probs = construct_category_probs(features, beta, link)
    return compute_mean_log_likelihood_from_category_probs(category_probs, labels)


def compute_mean_log_likelihood_from_category_probs(
    category_probs: NumpyArray2D, labels: NumpyArray2D
):
    choices = np.argmax(labels, 1)
    category_probs_of_choices = [
        category_probs[i, choice] for (i, choice) in enumerate(choices)
    ]
    return np.nanmean(np.log(category_probs_of_choices))


def generate_designed_features_and_multiclass_labels(
    features_non_autoregressive: NumpyArray2D,
    beta: NumpyArray2D,
    link: Link,
    is_autoregressive: bool = False,
) -> Tuple[NumpyArray2D]:
    """
    Returns:
        designed_features : an np.array of shape(n_samples, n_features_designed);
            The first `n_features_non_autoregressive` features are the input features
            The next `n_categories` features are the previous labels.
        labels:  an np.array of shape (n_samples, n_categories);
            each row is one-hot encoded
    """
    # TODO: Come up with a constructive name for features that don't include autoregressive features.

    if is_autoregressive:
        (
            designed_features,
            labels,
        ) = generate_designed_features_and_multiclass_labels_for_autoregressive_case(
            features_non_autoregressive,
            beta,
            link,
        )
    else:
        labels = generate_multiclass_labels_for_nonautoregressive_case(
            features_non_autoregressive,
            beta,
            link,
        )
        designed_features = features_non_autoregressive
    return designed_features, labels


def generate_multiclass_labels_for_nonautoregressive_case(
    features: NumpyArray2D,
    beta: NumpyArray2D,
    link: Link,
) -> NumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, num_beta_columns)
                num_beta_columns depends on the link function:
                    num_beta_columns = n_categories - 1 for Link.MULTI_LOGIT and Link.STICK_BREAKING
                    num_beta_columns = n_categories for Link.CBC_PROBIT
                Note that if the former is used with an extra beta weight, that column will just be ignored.
    Returns:
        labels:  an np.array of shape (n_samples, n_categories);
            each row is one-hot encoded
    """
    n_samples = np.shape(features)[0]
    probs = construct_category_probs(features, beta, link)
    n_categories = np.shape(probs)[1]
    labels = np.zeros((n_samples, n_categories), dtype=np.int)
    for i in range(n_samples):
        category_probs_for_sample = enforce_bounds_on_prob_vector(probs[i, :])
        labels[i] = np.random.multinomial(1, category_probs_for_sample)
    return labels


def generate_designed_features_and_multiclass_labels_for_autoregressive_case(
    features_non_autoregressive: NumpyArray2D,
    beta: NumpyArray2D,
    link: Link = Link.CBC_PROBIT,
) -> Tuple[NumpyArray2D]:
    """
    The function `generate_multiclass_labels_for_non_autoregressive_case` constructs all the category probs
    in advance, before seeing what labels are selected.  We cannot do that here.  In the autoregressive case,
    we need to know the previous label before we can give the category probabilities are.  Thus,
    we construct the pair (category_prob, label) one by one.

    Arguments:
        features_non_autoregressive: an np.array of shape (n_samples, n_features_non_autoregressive)
        beta: regression weights with shape (n_features_designed, num_beta_columns)
            The first n_features_non_autoregressive rows of beta are for the intercept (if used)
            and the external covariates.  The remaining rows govern transitions.
    Returns:
        designed_features : an np.array of shape(n_samples, n_features_designed);
            The first `n_features_non_autoregressive` features are the input features
            The next `n_categories` features are the previous labels.
        labels:  an np.array of shape (n_samples, n_categories);
            each row is one-hot encoded
    """
    if link != Link.CBC_PROBIT:
        raise NotImplementedError

    n_samples = np.shape(features_non_autoregressive)[0]
    n_features_non_autoregressive = np.shape(features_non_autoregressive)[1]
    n_entries_for_beta_per_category = np.shape(beta)[0]
    n_categories = np.shape(beta)[1]
    n_features_designed = n_features_non_autoregressive + n_categories

    if (
        not n_features_non_autoregressive + n_categories
        == n_entries_for_beta_per_category
    ):
        raise ValueError(
            f"This function expects the regression weight (beta) for each category "
            f"to have a number of entries which equals the number of non-autoregressive features (i.e. "
            f"external covariates + optional intercept) plus the number of categories."
        )

    warnings.warn(
        "Making a mock previous label which is an equal-sized proportion of all possible ones."
    )
    prev_label = np.ones(n_categories) / n_categories  # make a fake initial label

    labels = np.zeros((n_samples, n_categories), dtype=np.int)
    designed_features = np.zeros((n_samples, n_features_designed))
    for i in range(n_samples):
        designed_features_for_one_sample = np.hstack(
            (features_non_autoregressive[i, :], prev_label)
        )[np.newaxis, :]
        probs_for_sample = construct_category_probs(
            designed_features_for_one_sample, beta, link
        )[0]
        label = np.random.multinomial(1, probs_for_sample)
        labels[i, :] = label
        prev_label = label
        designed_features[i, :] = designed_features_for_one_sample
    return designed_features, labels


def compute_covariate_conditional_entropies_of_true_category_probabilities(
    covariates,
    beta_true,
    link,
):
    probs = construct_category_probs(covariates, beta_true, link)
    return scipy.stats.entropy(enforce_bounds_on_prob_vector(probs), axis=1)


def compute_mean_entropies_of_true_category_probabilities_by_label(
    covariates, labels, beta_true, link
):
    """
    This can be used to understand how much of the generative process is
    choosing various categories/labels due to "signal" (e.g. covariates + weights clearly preferring the category)
    versus "noise".   If entropy is low for a given category, it's more of a signal.  If entropy is high
    for a given category, it's more luck.

    To interpret a given entropy value, we could use Mike H's approach:  Generate discrete probability vectors
    with some subset near 0, and plot the entropy.
    """
    probs = construct_category_probs(covariates, beta_true, link)
    entropies = scipy.stats.entropy(probs, axis=1)
    choices = np.argmax(labels, 1)

    n_categories = np.shape(labels)[1]
    mean_entropies_by_label = {}
    for k in range(n_categories):
        mean_entropies_by_label[k] = np.round(np.mean(entropies[choices == k]), 3)
    return mean_entropies_by_label
