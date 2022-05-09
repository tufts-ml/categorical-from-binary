"""
Generate some binary regression data.
    + If the logistic link is used, this is logistic regression
    + If the probit link is used, this is probit regression
"""
from enum import Enum
from typing import List, NamedTuple, Optional

import numpy as np
import scipy.stats
from scipy.stats import norm

from categorical_from_binary.data_generation.util import construct_random_signs_as_integers
from categorical_from_binary.kl import sigmoid
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


class LinearPredictorScaleAllocator(Enum):
    """
    Control how a scale on the linear predictor is allocated to scales across the regression coefficients
    for each covariate.  We obtain the relevant formula by noticing that since entries of X and entries of
    beta are all independent with mean zero, then for each category Var(eta)=Var(X'beta)
    can be reduced to sum_m var(x_m) var(beta_m).  If we assume that the variance on the features is held
    fixed at 1.0, we have Var(eta) = sum_m var(beta_m).  In the function below, we fix
    sum_m var(x_m) = scale_for_linear_predictor, and this enum controls how `scale_for_linear_predictor`
    is allocated across the M covariates.

    Options:
        MAKE_UNIFORM_FEATURES:  For any given category, we control the
        variance of the linear predictor to be equal to be equally shared across the M covariates.  This can
        help prevent the linear predictors from blowing up for contexts where the number of covariates is large.
        In particular var(beta_m) = 1/m * scale_for_linear_predictor for all m and so
        sum_m var(beta_m) = scale_for_linear_predictor

        MAKE_PREFERRED_FEATURES:   Here we have
        var(beta_m) =  2 * scale_for_linear_predictor *m /  (M(M+1)),
            and so
        sum_m var(beta_m) = scale_for_linear_predictor.
    """

    MAKE_UNIFORM_FEATURES = 1
    MAKE_PREFERRED_FEATURES = 2


class Link(Enum):
    """
    Link function for the regression to binary responses
    """

    LOGISTIC = 1
    PROBIT = 2


class BinaryRegressionDataset(NamedTuple):
    features: np.ndarray  # shape:  (n_samples, n_features)
    labels: np.ndarray  # shape: (n_samples,)
    beta: np.ndarray  # shape: (n_features+1, ) ; 0th entry is the intercept term.
    link: Link


def generate_logistic_regression_dataset(
    n_samples: int,
    n_features: int,
    n_sparse_features: int = 0,
    n_bounded_from_zero_features: int = 0,
    lower_bound_for_bounded_from_zero_features: Optional[float] = None,
    mean: float = 0.0,
    scale_for_linear_predictor: float = 1.0,
    linear_predictor_scale_allocator: LinearPredictorScaleAllocator = LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES,
    mean_for_intercept: float = 0.0,
    scale_for_intercept: float = 0.25,
    seed: int = 1,
) -> BinaryRegressionDataset:
    return generate_binary_regression_dataset(
        n_samples,
        n_features,
        Link.LOGISTIC,
        n_sparse_features,
        n_bounded_from_zero_features,
        lower_bound_for_bounded_from_zero_features,
        mean,
        scale_for_linear_predictor,
        linear_predictor_scale_allocator,
        mean_for_intercept,
        scale_for_intercept,
        seed,
    )


def generate_probit_regression_dataset(
    n_samples: int,
    n_features: int,
    n_sparse_features: int = 0,
    n_bounded_from_zero_features: int = 0,
    lower_bound_for_bounded_from_zero_features: Optional[float] = None,
    mean: float = 0.0,
    scale_for_linear_predictor: float = 1.0,
    linear_predictor_scale_allocator: LinearPredictorScaleAllocator = LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES,
    mean_for_intercept: float = 0.0,
    scale_for_intercept: float = 0.25,
    seed: int = 1,
) -> BinaryRegressionDataset:
    return generate_binary_regression_dataset(
        n_samples,
        n_features,
        Link.PROBIT,
        n_sparse_features,
        n_bounded_from_zero_features,
        lower_bound_for_bounded_from_zero_features,
        mean,
        scale_for_linear_predictor,
        linear_predictor_scale_allocator,
        mean_for_intercept,
        scale_for_intercept,
        seed,
    )


def generate_binary_regression_dataset(
    n_samples: int,
    n_features: int,
    link: Link,
    n_sparse_features: int = 0,
    n_bounded_from_zero_features: int = 0,
    lower_bound_for_bounded_from_zero_features: Optional[float] = None,
    mean: float = 0.0,
    scale_for_linear_predictor: float = 1.0,
    linear_predictor_scale_allocator: LinearPredictorScaleAllocator = LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES,
    mean_for_intercept: float = 0.0,
    scale_for_intercept: float = 0.25,
    seed: int = 1,
) -> BinaryRegressionDataset:

    # TODO: extend purview of seed to reach beta_0 as well
    beta_0 = np.random.normal(loc=mean_for_intercept, scale=scale_for_intercept)
    beta_for_features = generate_regression_coefficients_for_features(
        n_features,
        n_sparse_features,
        n_bounded_from_zero_features,
        lower_bound_for_bounded_from_zero_features,
        mean,
        scale_for_linear_predictor,
        linear_predictor_scale_allocator,
        seed,
        positive_and_negative=True,
    )
    beta = np.insert(beta_for_features, 0, beta_0)

    features = generate_features_via_independent_normals(
        n_samples, n_features, mean=0.0, scale=1.0
    )
    labels = generate_labels_via_features_and_binary_regression_weights(
        features,
        beta,
        link,
    )
    return BinaryRegressionDataset(features, labels, beta, link)


def generate_regression_coefficients_for_features(
    n_features: int,
    n_sparse_features: int = 0,
    n_bounded_from_zero_features: int = 0,
    lower_bound_for_bounded_from_zero_features: Optional[float] = None,
    mean: float = 0.0,
    scale_for_linear_predictor: float = 1.0,
    linear_predictor_scale_allocator: LinearPredictorScaleAllocator = LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES,
    seed: int = 1,
    positive_and_negative: bool = True,
):
    """
    The features are ordered as:
        (all other features, bounded from zero features, sparse features)

    Arguments:
        linear_predictor_scale_allocator: Allocates how scale for linear predictor is allocated across the m covariats
    """
    if n_sparse_features + n_bounded_from_zero_features > n_features:
        raise ValueError(
            "The number of sparse features plus the number of "
            "bounded from zero features cannot exceed the total number of features."
        )
    np.random.seed(seed)
    n_regular_features = n_features - n_sparse_features - n_bounded_from_zero_features
    scales_for_regular_features = (
        np.ones(n_regular_features) * scale_for_linear_predictor
    )
    if (
        linear_predictor_scale_allocator
        == LinearPredictorScaleAllocator.MAKE_PREFERRED_FEATURES
    ):
        for r in range(n_regular_features):
            scales_for_regular_features[r] *= np.sqrt(
                2 * (r + 1) / ((n_regular_features) * (n_regular_features + 1))
            )
    elif (
        linear_predictor_scale_allocator
        == LinearPredictorScaleAllocator.MAKE_UNIFORM_FEATURES
    ):
        for r in range(n_regular_features):
            scales_for_regular_features[r] *= np.sqrt(1 / (n_regular_features))
    else:
        raise ValueError(
            "I am not sure how you want me to control the beta feature scale"
        )
    betas_regular = np.random.normal(
        loc=mean, scale=scales_for_regular_features, size=n_regular_features
    ).tolist()
    betas_bounded_from_zero_all_positive = np.array(
        [
            scipy.stats.truncnorm(
                a=lower_bound_for_bounded_from_zero_features,
                b=np.inf,
                loc=mean,
                scale=scale_for_linear_predictor,
            ).rvs()
            for i in range(n_bounded_from_zero_features)
        ]
    )
    if positive_and_negative:
        betas_bounded_from_zero = (
            betas_bounded_from_zero_all_positive
            * construct_random_signs_as_integers(n_bounded_from_zero_features)
        )
    else:
        betas_bounded_from_zero = betas_bounded_from_zero_all_positive
    betas_sparse = [0.0] * n_sparse_features
    beta = np.array(betas_regular + betas_bounded_from_zero.tolist() + betas_sparse)
    return beta


def generate_features_via_independent_normals(
    n_samples: int,
    n_features: int,
    mean: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Returns:
        features, an np.array of shape (n_samples, n_features)
    """
    features = np.zeros((n_samples, n_features), dtype=np.float)
    for i in range(n_samples):
        features_for_sample = np.random.normal(loc=mean, scale=scale, size=n_features)
        features[i, :] = features_for_sample
    return features


def generate_features(
    n_samples: int,
    n_features: int,
    mean: float = 0.0,
    scale: float = 1.0,
    bernoulli_indices: List[int] = None,
    bernoulli_probs: List[float] = None,
) -> np.ndarray:
    """
    Generates features from IID Gaussians, unless bernoulli_indices are specified.:

    Returns:
        features, an np.array of shape (n_samples, n_features)
    """
    if bernoulli_indices and max(bernoulli_indices) > n_features:
        raise ValueError(
            f"You've provided indices for bernoulli features that exceeds "
            f" the total number of specified features."
        )

    features = np.zeros((n_samples, n_features), dtype=np.float)

    # first pretend like all features are normal
    for i in range(n_samples):
        features_for_sample = np.random.normal(loc=mean, scale=scale, size=n_features)
        features[i, :] = features_for_sample

    # now overwrite the appropriate feature indices with binary variables
    if bernoulli_indices:
        for j, prob in zip(bernoulli_indices, bernoulli_probs):
            features[:, j] = np.random.binomial(1, prob, size=n_samples)
    return features


def generate_labels_via_features_and_binary_regression_weights(
    features: np.ndarray,
    beta: np.ndarray,
    link: Link,
) -> np.ndarray:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features)
        beta: regression weights with shape (n_features+1, )

    Returns:
        labels:  an np.array of shape (n_samples, ) and dtype int.
    """
    # TODO: Update this, and callers, to utilize the more streamlined functions
    # `construct_binary_logistic_probs` and `construct_binary_probit_probs`
    n_samples = np.shape(features)[0]
    labels = np.zeros((n_samples,), dtype=np.int)
    for i in range(n_samples):
        features_for_sample = features[i, :]
        linear_predictor = beta[0] + np.dot(beta[1:], features_for_sample)
        if link == Link.LOGISTIC:
            prob = sigmoid(linear_predictor)
        elif link == Link.PROBIT:
            prob = norm.cdf(linear_predictor)
        label = np.random.binomial(n=1, p=prob)
        labels[i] = label
    return labels


def construct_binary_logistic_probs(
    beta: NumpyArray1D, covariates: NumpyArray2D
) -> NumpyArray1D:
    linear_predictors = covariates @ beta
    return sigmoid(linear_predictors)


def construct_binary_probit_probs(
    beta: NumpyArray1D, covariates: NumpyArray2D
) -> NumpyArray1D:
    linear_predictors = covariates @ beta
    return norm.cdf(linear_predictors)
