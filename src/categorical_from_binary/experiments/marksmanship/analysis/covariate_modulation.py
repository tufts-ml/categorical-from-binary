import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np


np.set_printoptions(precision=3, suppress=True)

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression

from categorical_from_binary.covariate_dict import VariableType
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    CATEGORY_PROBABILITY_FUNCTION_BY_LINK,
    Link,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def _beta_matrix_from_linear_predictor(linear_predictor: NumpyArray1D) -> NumpyArray2D:
    """
    takes a linear predictor vector with shape (K,) and makes it a beta matrix
    with shape (1,K) - this lets us feed it into the category probability formulas
    """
    return linear_predictor[np.newaxis, :]


def compute_baseline_category_probs(
    covariate_dict, beta_mean, category_probability_link: Link
):
    index_intercept = covariate_dict["intercept"].index
    beta_matrix = _beta_matrix_from_linear_predictor(beta_mean[index_intercept, :])
    intercept_matrix = np.ones((1, 1))
    construct_category_probabilities = CATEGORY_PROBABILITY_FUNCTION_BY_LINK[
        category_probability_link
    ]
    return construct_category_probabilities(intercept_matrix, beta_matrix)


def compute_category_probs_after_adjusting_baseline_with_feature(
    covariate_dict,
    beta_mean: NumpyArray2D,
    feature_name: str,
    category_probability_link: Link,
    feature_z_score: float = 1.0,
    beta_stds: Optional[NumpyArray1D] = None,
    scalar_multiplier_on_beta_std_for_feature: float = 0.0,
) -> NumpyArray1D:
    """
    Notes:
        The arguments `beta_stds` and `scalar_multiplier_on_beta_std_for_feature` can be used
        to pull betas from the posterior distribution that aren't the posterior expectation.
        This can be used, for instance, to make confidence bands around the category probabilities,
        although I suspect that these computations are incorrect, and we would need to use the Jacobian
        from the change of variables formula to get the correct density.
    """
    index_intercept = covariate_dict["intercept"].index
    index_feature = covariate_dict[feature_name].index

    if beta_stds is None:
        beta_std_adjustment = 0.0
    else:
        warnings.warn(
            "This code is experimental and likely shouldn't be used; see function docstring"
        )
        beta_std_adjustment = (
            scalar_multiplier_on_beta_std_for_feature * beta_stds[index_feature],
        )

    linear_predictor = beta_mean[index_intercept, :] + feature_z_score * (
        beta_mean[index_feature, :] + beta_std_adjustment
    )
    beta_matrix = _beta_matrix_from_linear_predictor(linear_predictor)
    intercept_matrix = np.ones((1, 1))
    construct_category_probabilities = CATEGORY_PROBABILITY_FUNCTION_BY_LINK[
        category_probability_link
    ]
    # TODO: flatten the category probability function itself.
    return np.ndarray.flatten(
        construct_category_probabilities(intercept_matrix, beta_matrix)
    )


def compute_category_probs_after_adjusting_baseline_with_feature_for_sklearn(
    covariate_dict,
    feature_name: str,
    feature_z_score: float,
    sklearn_model: LogisticRegression,
) -> NumpyArray1D:
    """
    Assume the sklearn model `clf` was fit such that the 0th covariate was a "1" for the intercept term.
    Assume beta_sklearn's 0-th entry is for the intercept
    """

    num_predictors = len([v.index for v in covariate_dict.values()])
    index_feature = covariate_dict[feature_name].index

    covariate_test = np.zeros((1, num_predictors))
    index_of_fake_observation = 0
    covariate_test[index_of_fake_observation, 0] = 1
    covariate_test[index_of_fake_observation, index_feature] = feature_z_score
    return sklearn_model.predict_proba(covariate_test)[0]


###
# Used to make covariate modulation plots
###


def make_z_to_x(
    covariate_dict,
    feature_name: str,
) -> Callable:
    def z_to_x(z):
        mean, std = (
            covariate_dict[feature_name].raw_mean,
            covariate_dict[feature_name].raw_std,
        )
        return z * std + mean

    return z_to_x


def make_x_to_z(
    covariate_dict,
    feature_name: str,
) -> Callable:
    def x_to_z(x):
        mean, std = (
            covariate_dict[feature_name].raw_mean,
            covariate_dict[feature_name].raw_std,
        )
        return (x - mean) / std

    return x_to_z


class CategoryProbMakerForFeatureAtZScore(ABC):
    @abstractmethod
    def compute(self, feature_name: str, feature_z_score: float) -> NumpyArray1D:
        pass


class CategoryProbMakerForFeatureAtZScore_with_SKLEARN_model(
    CategoryProbMakerForFeatureAtZScore
):
    def __init__(self, covariate_dict, sklearn_model: LogisticRegression):
        self.covariate_dict = covariate_dict
        self.sklearn_model = sklearn_model

    def compute(self, feature_name: str, feature_z_score: float) -> NumpyArray1D:
        return compute_category_probs_after_adjusting_baseline_with_feature_for_sklearn(
            self.covariate_dict, feature_name, feature_z_score, self.sklearn_model
        )


class CategoryProbMakerForFeatureAtZScore_with_CAVI_model(
    CategoryProbMakerForFeatureAtZScore
):
    def __init__(
        self, covariate_dict, category_probability_link: Link, beta_mean: NumpyArray2D
    ):
        self.category_probability_link = category_probability_link
        self.beta_mean = beta_mean
        self.covariate_dict = covariate_dict

    def compute(self, feature_name: str, feature_z_score: float) -> NumpyArray1D:
        return compute_category_probs_after_adjusting_baseline_with_feature(
            self.covariate_dict,
            self.beta_mean,
            feature_name,
            self.category_probability_link,
            feature_z_score,
        )


def make_covariate_modulation_dict_for_feature_using_CAVI_model(
    covariate_dict,
    label_dict,
    beta_mean: NumpyArray2D,
    feature_name: str,
    category_probability_link: Link,
) -> pd.DataFrame:
    category_prob_maker_for_feature_at_z_score = (
        CategoryProbMakerForFeatureAtZScore_with_CAVI_model(
            covariate_dict, category_probability_link, beta_mean
        )
    )
    return _make_covariate_modulation_dict(
        covariate_dict,
        label_dict,
        feature_name,
        category_prob_maker_for_feature_at_z_score,
    )


def make_covariate_modulation_dict_for_feature_using_SKLEARN_model(
    covariate_dict,
    label_dict,
    sklearn_model: LogisticRegression,
    feature_name: str,
) -> pd.DataFrame:
    category_prob_maker_for_feature_at_z_score = (
        CategoryProbMakerForFeatureAtZScore_with_SKLEARN_model(
            covariate_dict, sklearn_model
        )
    )
    return _make_covariate_modulation_dict(
        covariate_dict,
        label_dict,
        feature_name,
        category_prob_maker_for_feature_at_z_score,
    )


def _make_covariate_modulation_dict(
    covariate_dict,
    label_dict,
    feature_name: str,
    category_prob_maker_for_feature_at_z_score: CategoryProbMakerForFeatureAtZScore,
) -> pd.DataFrame:
    """
    Has columns "covariate", "z-score", "prob" (i.e. category probability), "response" (i.e. category label)
    """

    x_to_z = make_x_to_z(covariate_dict, feature_name)
    z_to_x = make_z_to_x(covariate_dict, feature_name)

    z_min, z_max = (
        x_to_z(covariate_dict[feature_name].raw_min),
        x_to_z(covariate_dict[feature_name].raw_max),
    )
    is_binary = covariate_dict[feature_name].type.name == "BINARY"
    if is_binary:
        n_samples_from_zscore_space = 2
    else:
        n_samples_from_zscore_space = 101

    feature_z_scores = np.linspace(z_min, z_max, n_samples_from_zscore_space)

    category_probs_by_z_score = {}
    for feature_z_score in feature_z_scores:
        category_probs_after_feature_adjustment = (
            category_prob_maker_for_feature_at_z_score.compute(
                feature_name,
                feature_z_score,
            )
        )
        category_probs_by_z_score[
            feature_z_score
        ] = category_probs_after_feature_adjustment

    # now put this in dataframe form for plotting
    covariate_modulation_dict = {
        "covariate": [],
        "z-score": [],
        "prob": [],
        "response": [],
    }

    for label, label_info in label_dict.items():
        label_info.index
        covariate_modulation_dict["z-score"].extend(feature_z_scores)
        covariate_modulation_dict["covariate"].extend(z_to_x(feature_z_scores))

        curr_category_prob_by_z_score = [
            category_probs[label_info.index]
            for category_probs in category_probs_by_z_score.values()
        ]
        covariate_modulation_dict["prob"].extend(curr_category_prob_by_z_score)

        covariate_modulation_dict["response"].extend([label] * len(feature_z_scores))

    return pd.DataFrame(covariate_modulation_dict)


def print_report_on_binary_features(
    covariate_dict, beta_mean, category_probability_link: Link
):
    binary_feature_names = [
        name
        for name, info in covariate_dict.items()
        if info.type == VariableType.BINARY
    ]
    for feature_name in binary_feature_names:
        category_probs_baseline = compute_baseline_category_probs(
            covariate_dict, beta_mean, category_probability_link
        )
        category_probs_adjusted = (
            compute_category_probs_after_adjusting_baseline_with_feature(
                covariate_dict,
                beta_mean,
                feature_name,
                category_probability_link,
            )
        )
        print(f"--------")
        print(f"Baseline category probs: \n{category_probs_baseline}")
        print(
            f"\nAfter adding the binary feature called {feature_name}, controlling for other covariates: \n{category_probs_adjusted}"
        )


def print_report_on_continuous_features(
    covariate_dict, beta_mean, category_probability_link: Link
):
    continuous_feature_names = [
        name
        for name, info in covariate_dict.items()
        if info.type == VariableType.CONTINUOUS and name != "intercept"
    ]
    for feature_name in continuous_feature_names:
        category_probs_baseline = compute_baseline_category_probs(
            covariate_dict, beta_mean, category_probability_link
        )
        category_probs_plus_one_std = (
            compute_category_probs_after_adjusting_baseline_with_feature(
                covariate_dict,
                beta_mean,
                feature_name,
                category_probability_link,
            )
        )
        category_probs_minus_one_std = (
            compute_category_probs_after_adjusting_baseline_with_feature(
                covariate_dict,
                beta_mean,
                feature_name,
                category_probability_link,
                feature_z_score=-1,
            )
        )
        print(f"--------")
        print(
            f"\nCategory probs after subtracting 1 std from {feature_name}, controlling for other covariates: \n{category_probs_minus_one_std}"
        )
        print(f"\nBaseline category probs: \n{category_probs_baseline}")
        print(
            f"\nCategory probs after adding 1 std to {feature_name}, controlling for other covariates: \n{category_probs_plus_one_std}"
        )


def construct_dataframe_showing_covariate_modulation_for_binary_features(
    covariate_dict,
    label_dict,
    beta_mean,
    category_probability_link: Link,
    binary_feature_names_to_use: Optional[List[str]] = None,
) -> DataFrame:
    """
    Arguments:
        binary_feature_names_to_use : Optional. If None, we use all binary feature names
    """
    if binary_feature_names_to_use is None:
        binary_feature_names = [
            name
            for name, info in covariate_dict.items()
            if info.type == VariableType.BINARY
        ]
    else:
        binary_feature_names = binary_feature_names_to_use

    label_names = list(label_dict.keys())
    n_columns = len(binary_feature_names) + 1
    n_rows = len(label_names)
    response_probs_by_binary_covariates = np.zeros((n_rows, n_columns))

    category_probs_baseline = compute_baseline_category_probs(
        covariate_dict, beta_mean, category_probability_link
    )
    response_probs_by_binary_covariates[:, 0] = category_probs_baseline

    for (j, feature_name) in enumerate(binary_feature_names):
        category_probs_adjusted = (
            compute_category_probs_after_adjusting_baseline_with_feature(
                covariate_dict,
                beta_mean,
                feature_name,
                category_probability_link,
            )
        )
        response_probs_by_binary_covariates[:, j + 1] = category_probs_adjusted

    return pd.DataFrame(
        data=response_probs_by_binary_covariates,
        index=label_names,
        columns=["Baseline"] + binary_feature_names,
    )
