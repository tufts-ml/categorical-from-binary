import numpy as np


np.set_printoptions(suppress=True, precision=3)

from dataclasses import dataclass
from typing import Union

from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


@dataclass
class EvaluationResult:
    """
    false positive rate, true positive rate, accuracy
    """

    fpr: float
    tpr: float
    acc: float


def get_evaluation_result(
    beta_true: Union[NumpyArray1D, NumpyArray2D],
    model_inclusion_decision: Union[NumpyArray1D, NumpyArray2D, DataFrame],
):
    if np.ndim(beta_true) == 1:
        return get_evaluation_result_for_binary_data(
            beta_true, model_inclusion_decision
        )
    elif np.ndim(beta_true) == 2:
        return get_evaluation_result_for_multiclass_data(
            beta_true, model_inclusion_decision
        )
    else:
        raise ValueError(
            "Not sure how to handle the dimensionality of the provided true beta"
        )


def get_evaluation_result_for_multiclass_data(
    beta_true: NumpyArray2D,
    model_inclusion_decision: Union[NumpyArray2D, DataFrame],
) -> EvaluationResult:
    """
    Gets tpr, fpr, and accuracy

    Arguments:
        model_inclusion_decision: array (possibly data of a DataFrame) with shape (M,K), where M is number of covariates and K is
        number of categories
    """
    if isinstance(model_inclusion_decision, DataFrame):
        model_inclusion_decision = model_inclusion_decision.to_numpy()

    m, k = np.shape(beta_true)
    s_by_category = np.sum(
        beta_true == 0.0, axis=0
    )  # sparsity by category; assumed to be the last s entries
    if (s_by_category == s_by_category[0]).all():
        s = s_by_category[0]
    else:
        raise ValueError(
            "This function assumes the true sparsity pattern is identical across categories"
        )

    true_inclusion_decision = np.vstack((np.ones((m - s, k)), np.zeros((s, k))))

    false_positive_rate = np.sum(
        np.logical_and(model_inclusion_decision == 1, true_inclusion_decision == 0)
    ) / (np.sum(true_inclusion_decision == 0))
    true_positive_rate = np.sum(
        np.logical_and(model_inclusion_decision == 1, true_inclusion_decision == 1)
    ) / (np.sum(true_inclusion_decision == 1))
    accuracy = np.mean(model_inclusion_decision == true_inclusion_decision)
    return EvaluationResult(false_positive_rate, true_positive_rate, accuracy)


def get_evaluation_result_for_binary_data(
    beta_true: NumpyArray1D,
    model_inclusion_decision: Union[NumpyArray1D, DataFrame],
) -> EvaluationResult:
    """
    Warning:
        Has only been used on binary data so far.

    Arguments:
        model_inclusion_decision: DataFrame whose data has shape (M,), where M is number of covariates
    """
    if np.ndim(beta_true) != 1:
        raise NotImplementedError(
            "So far, this fucntion has only been developed and applied to binary models"
        )

    if isinstance(model_inclusion_decision, DataFrame):
        model_inclusion_decision = model_inclusion_decision.to_numpy()

    p = np.shape(beta_true)[0]  # number of predictors
    s = np.sum(beta_true == 0.0)  # sparsity; assumed to be the last s entries
    true_inclusion_decision = np.hstack((np.ones(p - s), np.zeros(s)))

    false_positive_rate = np.sum(
        np.logical_and(model_inclusion_decision == 1, true_inclusion_decision == 0)
    ) / (np.sum(true_inclusion_decision == 0))
    true_positive_rate = np.sum(
        np.logical_and(model_inclusion_decision == 1, true_inclusion_decision == 1)
    ) / (np.sum(true_inclusion_decision == 1))
    accuracy = np.mean(model_inclusion_decision == true_inclusion_decision)
    return EvaluationResult(false_positive_rate, true_positive_rate, accuracy)


def print_report_on_variable_selection_decisions(
    X: NumpyArray2D,
    y: NumpyArray1D,
    beta_mean: NumpyArray1D,
    beta_variational_stds: NumpyArray1D,
    beta_true: NumpyArray1D,
    feature_inclusion_df: DataFrame,
    sklearn_logistic_lasso_regression_C: float = 1.0,
    verbose: bool = False,
):
    """
    Addresses the question:
        how does our variatinal inference scheme compare to scikit's logistic lasso regression in terms
        of making variable selection decisions?

    Arguments:
        feature_inclusion_df: has shape (M,K), where M is number of covariates and K is the
            number of categories
    """
    # TODO: Align with `get_evaluation_result`

    if np.ndim(beta_true) != 1:
        raise NotImplementedError(
            "So far, this fucntion has only been developed and applied to binary models"
        )

    p = np.shape(beta_true)[0]  # number of predictors

    variational_inclusion_decision = feature_inclusion_df

    ### Get variable selection decisions for logistic regression, possibly with lasso
    # TODO: the sklearn fucntion should be factored out of this code; I have similar code for sklearn
    # within the marksmanship subpackage.
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        random_state=0,
        C=sklearn_logistic_lasso_regression_C,
    ).fit(X, y)
    scikit_beta = np.matrix.flatten(np.vstack((clf.intercept_, clf.coef_.T)))
    scikit_inclusion_decision = np.array([int(x) for x in abs(scikit_beta) > 0.0])

    info = np.hstack(
        (
            beta_true[:, np.newaxis],
            scikit_beta[:, np.newaxis],
            scikit_inclusion_decision[:, np.newaxis],
            beta_mean[:, np.newaxis],
            beta_variational_stds[:, np.newaxis],
            variational_inclusion_decision,
        )
    )
    if verbose:
        print(
            f"\nBeta true, scikit beta, scikit inclusion decision, variational mean beta, variational beta stds, variational inclusion decision: \n {info}"
        )

    ### get accuracy in variable selection decisions
    s = np.sum(beta_true == 0.0)  # sparsity; assumed to be the last s entries
    true_inclusion_decision = np.hstack((np.ones(p - s), np.zeros(s)))
    scikit_inclusion_accuracy = np.mean(
        scikit_inclusion_decision == true_inclusion_decision
    )
    variational_inclusion_accuracy = np.mean(
        variational_inclusion_decision == true_inclusion_decision[:, np.newaxis]
    )[0]
    print(
        f"Inclusion decision accuracy -- scikit: {scikit_inclusion_accuracy:.02}, variational: {variational_inclusion_accuracy:.02}"
    )
