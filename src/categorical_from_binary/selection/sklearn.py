from typing import Dict, Optional

import numpy as np


np.set_printoptions(precision=3, suppress=True)

from dataclasses import dataclass

import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.selection.evaluate import (
    EvaluationResult,
    get_evaluation_result,
)
from categorical_from_binary.sklearn import (
    fit_sklearn_binary_logistic_regression,
    fit_sklearn_multiclass_logistic_regression,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


@dataclass
class SklearnVariableSelectionResults:
    beta: NumpyArray1D
    feature_inclusion_df: DataFrame


def get_sklearn_beta_vector_from_logistic_regression(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    sklearn_logistic_regression_C: float,
    multiclass: bool,
    penalty: str,
    solver: Optional[str] = None,
) -> NumpyArray1D:
    """
    `features` should NOT include a column of ones for intercept
    """
    # TODO: I could probably rewrite my stacking operations so that
    # the return statement is the same for both cases and lies outside of the if/else.
    if multiclass:
        sklearn_model = fit_sklearn_multiclass_logistic_regression(
            features, labels, sklearn_logistic_regression_C, penalty, solver
        )
        if sklearn_model.fit_intercept:
            return np.insert(sklearn_model.coef_.T, 0, sklearn_model.intercept_, axis=0)
        else:
            return sklearn_model.coef_.T
    else:
        sklearn_model = fit_sklearn_binary_logistic_regression(
            features,
            labels,
            sklearn_logistic_regression_C,
            penalty,
            solver,
        )
        return np.matrix.flatten(
            np.vstack((sklearn_model.intercept_, sklearn_model.coef_.T))
        )


def get_sklearn_variable_selection_results_using_logistic_regression(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    sklearn_logistic_regression_C: float,
    multiclass: bool,
    penalty: str,
    solver: Optional[str] = None,
) -> SklearnVariableSelectionResults:
    """
    `features` should NOT include a column of ones for intercept
    """
    sklearn_beta = get_sklearn_beta_vector_from_logistic_regression(
        features,
        labels,
        sklearn_logistic_regression_C,
        multiclass,
        penalty,
        solver,
    )
    sklearn_feature_inclusion_matrix = np.array(abs(sklearn_beta) > 0.0, dtype=int)
    sklearn_feature_inclusion_df = pd.DataFrame(
        data=sklearn_feature_inclusion_matrix,
    )
    return SklearnVariableSelectionResults(sklearn_beta, sklearn_feature_inclusion_df)


def get_evaluation_results_by_sklearn_Cs(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    beta_true: NumpyArray1D,
    sklearn_logistic_lasso_regression_Cs: NumpyArray1D,
    multiclass: bool,
) -> Dict[float, EvaluationResult]:
    """
    `features` should NOT include a column of ones for intercept
    Arguments:
        scikit_inclusion_decision: the return value of `get_sklearn_inclusion_decision_matrix_from_logistic_lasso_regression`
    """
    # if multiclass:
    #     raise NotImplementedError

    evaluation_results_by_sklearn_Cs = {}
    for sklearn_logistic_lasso_regression_C in sklearn_logistic_lasso_regression_Cs:
        print(
            f"Now evaluating the variable selection decisions of sklearn with C value of {sklearn_logistic_lasso_regression_C:.03f}"
        )
        sklearn_feature_inclusion_df = (
            get_sklearn_variable_selection_results_using_logistic_regression(
                features,
                labels,
                sklearn_logistic_lasso_regression_C,
                multiclass,
                penalty="l1",
            ).feature_inclusion_df
        )
        evaluation_result_sklearn = get_evaluation_result(
            beta_true, sklearn_feature_inclusion_df
        )
        evaluation_results_by_sklearn_Cs[
            sklearn_logistic_lasso_regression_C
        ] = evaluation_result_sklearn
    return evaluation_results_by_sklearn_Cs
