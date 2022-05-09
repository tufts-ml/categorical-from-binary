from typing import Optional

import numpy as np


np.set_printoptions(precision=3, suppress=True)

from sklearn.linear_model import LogisticRegression

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def fit_sklearn_binary_logistic_regression(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    sklearn_logistic_regression_C: Optional[float],
    penalty: str,
    solver: Optional[str] = None,
) -> LogisticRegression:
    """
    `features` should NOT include a column of ones for intercept
    """

    if (features[:, 0] == 1).all():
        raise ValueError(
            "The features should not include a column of all 1's for the intercept; "
            "sklearn will add on the intercept AND not penalize it."
        )

    if penalty == "l2" and solver is None:
        solver = "newton-cg"
    elif penalty == "l1" and solver is None:
        solver = "liblinear"
    elif solver is None:
        raise ValueError(
            "Not sure what to use as default solver, consult sklearn for guidance"
        )

    X = features
    y = labels

    clf = LogisticRegression(
        penalty=penalty,
        solver=solver,
        random_state=0,
        C=sklearn_logistic_regression_C,
    ).fit(X, y)
    return clf


def fit_sklearn_multiclass_logistic_regression(
    covariates: NumpyArray2D,
    labels: NumpyArray2D,
    sklearn_logistic_regression_C: float = 1.0,
    penalty: str = "l2",
    solver: Optional[str] = None,
) -> LogisticRegression:

    X = covariates
    y = np.argmax(labels, 1)

    if penalty == "l2" and solver is None:
        solver = "newton-cg"
    elif penalty == "l1" and solver is None:
        solver = "saga"
    elif solver is None:
        raise ValueError(
            "Not sure what to use as default solver, consult sklearn for guidance"
        )

    ### Get variable selection decisions for logistic regression, possibly with lasso
    clf = LogisticRegression(
        penalty=penalty,
        solver=solver,
        random_state=0,
        multi_class="multinomial",
        C=sklearn_logistic_regression_C,
        fit_intercept=False,
    ).fit(X, y)
    return clf
