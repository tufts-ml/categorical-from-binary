from typing import Dict, Optional

import numpy as np

from categorical_from_binary.ib_cavi.binary_probit.inference.main import (
    compute_probit_vi_with_normal_gamma_prior,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_gamma_prior,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.util import beta_stds_from_beta_cov
from categorical_from_binary.selection.core import (
    compute_feature_inclusion_data_frame_using_scaled_neighborhood_method,
)
from categorical_from_binary.selection.evaluate import (
    EvaluationResult,
    get_evaluation_result,
)
from categorical_from_binary.selection.hyperparameters import (
    hyperparameters_from_lambda_and_desired_marginal_beta_variance,
)
from categorical_from_binary.types import NumpyArray1D


###
# feature selection accuracy for vb
###


def get_evaluation_results_for_variational_bayes_by_lambdas_(
    labels,
    covariates,
    beta_true: NumpyArray1D,
    multiclass: bool,
    max_n_iterations: int = 10,
    lambdas_: Optional[NumpyArray1D] = None,
) -> Dict[float, EvaluationResult]:
    """
    Function to find a good hyperparameter lambda_ for variable selection when applying CAVI to the
    binary probit model.   Note that this function holds three other hyperparmeters fixed [the variance
    of beta (another hyperparameter), plus two variable selection related hyperparameters).

    Usage:

        # evaluation results for variational bayes model
        lambdas_ = np.logspace(start=-5, stop=2.5, num=15, base=10)
        evaluation_results_by_lambdas_ = get_evaluation_results_for_variational_bayes_by_lambdas_(
            labels, covariates, dataset.beta, lambdas_
        )

        # evaluation results with sklearn
        from categorical_from_binary.selection.sklearn import get_evaluation_results_by_sklearn_Cs
        sklearn_logistic_lasso_regression_Cs = np.logspace(start=-2, stop=0, num=24, base=10)
        evaluation_results_by_sklearn_Cs = get_evaluation_results_by_sklearn_Cs(
            dataset.features, dataset.labels, dataset.beta, sklearn_logistic_lasso_regression_Cs
        )

    Notes:
        `features` should NOT include a column of ones for intercept
    Arguments:
        scikit_inclusion_decision: the return value of `get_sklearn_inclusion_decision_matrix_from_logistic_lasso_regression`
    """

    if lambdas_ is None:
        lambdas_ = np.logspace(start=-5, stop=2.5, num=15, base=10)

    evaluation_results_by_lambdas_ = {}

    for lambda_ in lambdas_:
        print(
            f"Now evaluating the variable selection decisions of CAVI with lambda_ value of {lambda_:.03f}"
        )
        variance = 1.0
        hyperparameters = (
            hyperparameters_from_lambda_and_desired_marginal_beta_variance(
                variance, lambda_
            )
        )
        if multiclass:
            inference_function = compute_multiclass_probit_vi_with_normal_gamma_prior
        else:
            inference_function = compute_probit_vi_with_normal_gamma_prior

        variational_params = inference_function(
            labels,
            covariates,
            variational_params_init=None,
            max_n_iterations=max_n_iterations,
            convergence_criterion_drop_in_elbo=-np.inf,
            hyperparameters=hyperparameters,
        )

        # TODO: Make this a function argument, don't hardcode it.
        neighborhood_probability_threshold_for_exclusion = 0.90
        neighborhood_radius_in_units_of_std_devs = 3

        beta_mean = variational_params.beta.mean
        beta_stds = beta_stds_from_beta_cov(variational_params.beta.cov)

        feature_inclusion_df = (
            compute_feature_inclusion_data_frame_using_scaled_neighborhood_method(
                beta_mean,
                beta_stds,
                neighborhood_probability_threshold_for_exclusion,
                neighborhood_radius_in_units_of_std_devs,
            )
        )
        # TODO: if this stops working in the binary case, change the code so that it does the
        # following below in the binary case.
        #
        # variational_inclusion_decision = np.matrix.flatten(
        #    feature_inclusion_df.to_numpy()
        # )
        evaluation_result_vb = get_evaluation_result(beta_true, feature_inclusion_df)
        evaluation_results_by_lambdas_[lambda_] = evaluation_result_vb

    return evaluation_results_by_lambdas_
