"""
Generate some hierarchical multiclass regression data 
"""
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from categorical_from_binary.data_generation.bayes_binary_reg import generate_features
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    MulticlassRegressionDataset,
    generate_designed_features_and_multiclass_labels,
)
from categorical_from_binary.data_generation.design import Design
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


# TODO: Move 'types' to more central place than simple_hdpmm

# TODO: The logistic regression modules defines 'features' without the all ones column;
# This module defines 'features' as having the ones column.  Make it consistent across modules.


@dataclass
class HierarchicalMulticlassRegressionDataset:
    """
    Aka Hierarchical Polytomous Regression Dataset

    Attributes:

        datasets: List[MulticlassRegressionDataset]
            The length of the list is n_groups
            Each group has their own features, betas, labels.
        beta_expected: np.ndarray  # shape: (num_designed_features, num_beta_columns)
            Expected regression weights across groups.
            num_beta_columns depends on the link function:
                num_beta_columns = n_categories - 1 for Link.MULTI_LOGIT and Link.STICK_BREAKING
                num_beta_columns = n_categories for Link.CBC_PROBIT
            num_designed_features is given by
                num_designed_features = n_features, if there is no intercept added
                num_designed_features = n_features + 1, if there is an intercept added
            This is optional because for a real dataset it won't be known.
        beta_cov: np.ndarray # shape : (num_designed_features, num_designed_features)
            Covariance in regression weights across groups.
            Assumed to be constant across the n_categories.
            This is optional because for a real dataset it won't be known.
        link: Link
            The link function used to generate the responses from the features
            and beta. Important for interpreting the beta coefficients.
            This is currently also redundantly stored within each group's dataset... yuck.
            This is optional because for a real dataset it won't be known.
        seed: int
            The numpy seed used for the function
            This is currently also redundantly stored within each group's dataset... yuck.
            This is optional because for a real dataset it won't be known.
        is autoregressive : bool
            If true, the category at timestep t-1 was used as a predictor for the category
            at time t.   Note that these predictors are NOT stored in
            MulticlassRegressionDataset.features
            This is optional because for a real dataset it won't be known.
    """

    datasets: List[MulticlassRegressionDataset]
    beta_expected: Optional[NumpyArray2D] = None
    beta_cov: Optional[NumpyArray2D] = None
    link: Optional[Link] = None
    seed: Optional[int] = None
    is_autoregressive: Optional[bool] = None


def generate_hierarchical_multiclass_regression_dataset(
    n_samples: int,
    n_features_exogenous: int,
    n_categories: int,
    n_groups: int,
    s2_beta: float = 1,
    s2_beta_expected: float = 1,
    link: Link = Link.CBC_PROBIT,
    seed: int = None,
    include_intercept: bool = True,
    beta_0_expected: Optional[NumpyArray1D] = None,
    is_autoregressive: bool = False,
    beta_transition_expected: Optional[NumpyArray2D] = None,
    beta_exogenous_expected: Optional[NumpyArray2D] = None,
    indices_of_exogenous_features_that_are_binary: Optional[List[int]] = None,
    success_probs_for_exogenous_features_that_are_binary: Optional[List[float]] = None,
) -> HierarchicalMulticlassRegressionDataset:
    """
    Generate multiclass regression data in the presence of `n_groups` groups.

    We first compute expected betas (where the expectation is taken across groups).  Recall that
    in this setting, betas have dimension (n_designed_features, n_categories).  See
    `get_num_features_designed` for how this is computed

    We then sample each group's beta given the expected betas above, and a covariance matrix that is
    constant across groups, and set to be s2*I, for some scalar s2, and where I has dimesionality
    (n_designed_features, n_designed_features).  Note that as s2 increases, the regression weights become
    more dissimilar across groups.

    Finally, we can then generate a bunch of group-specific multiclass regression data in the same way
    as we do in the non-hierarchical setting (see `generate_multiclass_regression_dataset`)

    Arguments:
        n_features_exogenous:
            This is the number of features or predictors, not including the intercept or the transition feature
        s2_beta_expected:
            The expected beta for the m-th covariate in the design matrix and k-th category is sampled from
                N(0, s2_beta_expected)
            So higher values of s2_beta_expected makes the regression weights more variable across
            covariates and categories (and therefore increases the tendency of those covariates to "matter"
            in determining the category probabilities).  Note that this effect can be partially overriden
            if one provides `beta_0_expected` and/or `beta_transition_expected`.
        s2_beta:
            For any response category (k), betas for each group are sampled from
                N(beta_expected_for_response_category_k, Sigma_k)
            And we set Sigma_k := s2_beta * I.
            So higher values of s2_beta makes the regression weights more variable across groups;
            lower values of s2_beta make regression weights more similar across groups.
        link:
            The link function used for the multiclass logistic regression.  Note that the link function
            determines the dimensionality of beta
        include_intercept:
            If True, the 0-th row of the beta matrix will correspond to the intercept, and the 0-th column
            of the features matrix will be a column of all 1's.  If False, neither condition will be true.
        beta_0_expected:
            Optional.  If present, include_intercept must be True.
            If present, it's a numpy array of size (K,),  where K is the number of categories.
            When present, we don't simulate beta_0_expected randomly, but use the provided value instead.
            `beta_0_expected` is the expected intercept across all the groups.
        is_autoregressive:
            If True,
            Note that if True, one does not need include_intercept to be True.
        beta_transition_expected:
            Optional. If present, is_autoregressive must be True.
            Allows one to overwrite the bottom KxK block of beta_expected so that we can control the
            influence of categories on each other (e.g., we can encourage self-transitions.)  Note that if
            we use this, we likely want to also diminish the influence of external covariates by reducing
            the size of s2_beta_expected.
        beta_exogenous_expected:
            Optional.  Allows one to overwrite the (n_features_exogenous X K) block of beta_expected so
            that we can control the influence of exogenous covariates on category probabilities.
        indices_of_exogenous_features_that_are_binary:
            Optional.  If present, the exogenous features with the provided indices
            (with ZERO-INDEXING and in EXOGENOUS FEATURE COORDINATES, i.e.
            from i=0,...,n_features_exogenous-1) are made to be binary, with success probabilities
            given by `success_probs_for_exogenous_features_that_are_binary`
    """

    ###
    # Generate expected (across groups) beta coefficients
    ###
    np.random.seed(seed)
    if link != Link.CBC_PROBIT:
        raise NotImplementedError(
            f"Only Link.CBC_PROBIT is currently supported. For example, Link.STICK_BREAKING "
            f"and Link.MULTI_LOGIT use one less column for beta than the number of categories. This function needs "
            f"to be adjusted to handle that.  For guidance, see the data generation model for the non-hierarchical case."
        )
    if beta_0_expected is not None and include_intercept == False:
        raise ValueError(
            f"You have provided a `beta_0_expected` but don't want to include an intercept.  WTF, man?"
        )
    if beta_transition_expected is not None and is_autoregressive == False:
        raise ValueError(
            f"You have provided a ` beta_transition_expected` but don't want to include "
            f"simulate data autoregressively.  WTF, man?"
        )
    if (
        beta_exogenous_expected is not None
        and np.shape(beta_exogenous_expected)[0] != n_features_exogenous
    ):
        raise ValueError(
            f"You have specified expected betas for the exogenous covariates, but the number of rows "
            f"does not match what you have specified in `n_features_exogenous`."
        )
    if include_intercept and is_autoregressive:
        warnings.warn(
            f"You have turned on both `include_intercept` and `is_autoregressive`.  Is this "
            f"definitely what you want?  If the latter is True, there is no longer a need for the former. "
            f"On the other hand, apparently Bayesians can use non-identified models, and the CBC-Probit "
            f"is non-identified anyways."
        )
    if indices_of_exogenous_features_that_are_binary is None:
        indices_of_exogenous_features_that_are_binary = []
    if success_probs_for_exogenous_features_that_are_binary is None:
        success_probs_for_exogenous_features_that_are_binary = []

    design = Design(
        n_features_exogenous, include_intercept, n_categories * is_autoregressive
    )

    # TODO: Allow control over the relative influence of category probabilities
    beta_expected = np.random.normal(
        loc=0, scale=s2_beta_expected, size=(design.num_features_designed, n_categories)
    )

    # Overwrite beta expected with desired terms.
    if beta_0_expected is not None:
        beta_expected[0, :] = beta_0_expected
    if beta_transition_expected is not None:
        beta_expected[-n_categories:, -n_categories:] = beta_transition_expected
    if beta_exogenous_expected is not None:
        f, l = design.index_bounds_exogenous_features
        beta_expected[f:l, :] = beta_exogenous_expected

    # TODO: Add option to provide beta_transition_expected, a KxK np.array, which will allow us
    # to favor self transitions.  Note that to not have this favoring outweighed by the exogenous
    # features or the intercept, we will need to check our provided values against the variance
    # from those blocks.

    ###
    # Generate betas across group and categories
    ###
    beta_cov = s2_beta * np.eye(design.num_features_designed)
    betas_across_groups_and_categories = np.zeros(
        (n_groups, design.num_features_designed, n_categories)
    )
    for k in range(n_categories):
        betas_across_groups_and_categories[:, :, k] = np.random.multivariate_normal(
            mean=beta_expected[:, k], cov=beta_cov, size=n_groups
        )

    ###
    # Construct dataset (betas, features, labels) for each group
    ###

    datasets = [None] * n_groups

    for j in range(n_groups):

        # generate non autoregressive features
        features_exogenous = generate_features(
            n_samples,
            design.num_features_exogenous,
            mean=0.0,
            scale=1.0,
            bernoulli_indices=indices_of_exogenous_features_that_are_binary,
            bernoulli_probs=success_probs_for_exogenous_features_that_are_binary,
        )

        if include_intercept:
            features_non_autoregressive = (
                prepend_features_with_column_of_all_ones_for_intercept(
                    features_exogenous
                )
            )
        else:
            features_non_autoregressive = features_exogenous

        beta_for_group = betas_across_groups_and_categories[j, :, :]
        designed_features, labels = generate_designed_features_and_multiclass_labels(
            features_non_autoregressive,
            beta_for_group,
            link,
            is_autoregressive,
        )

        datasets[j] = MulticlassRegressionDataset(
            designed_features, labels, beta_for_group, link, seed
        )

    return HierarchicalMulticlassRegressionDataset(
        datasets, beta_expected, beta_cov, link, seed, is_autoregressive
    )
