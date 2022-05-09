from typing import Dict

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_cbc_logit_probabilities,
    construct_cbc_probit_probabilities,
    construct_cbm_logit_probabilities,
    construct_cbm_probit_probabilities,
    construct_non_identified_softmax_probabilities,
)
from categorical_from_binary.hmc.core import CategoricalModelType


CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE = {
    CategoricalModelType.CBC_PROBIT: construct_cbc_probit_probabilities,
    CategoricalModelType.CBM_PROBIT: construct_cbm_probit_probabilities,
    CategoricalModelType.CBC_LOGIT: construct_cbc_logit_probabilities,
    CategoricalModelType.CBM_LOGIT: construct_cbm_logit_probabilities,
    CategoricalModelType.SOFTMAX: construct_non_identified_softmax_probabilities,
}


def report_on_intercepts_only_experiment(
    true_category_probs_K: np.array,
    categorical_model_type: CategoricalModelType,
    betas_SKM_by_N: Dict[int, np.array],
):
    """
    Print a table which checks the inferred category frequencies
    when plugging the regression weights into the {CBC, CBM} category probability functions.

    Note that there is no notion of training or test data here, since this is intercepts-only data.

    Arguments:
        categorical_model_type: the type of categorical model used when determining the regression weights (betas)
            provided in argument `betas_SKM_by_N`
        betas_SKM_by_N : dict mapping number of samples (a subset of the training data) to sampled betas;
            the sampled betas have shape (S,K), where S is the number of MCMC samples, K
            is the number of categories, and M is the dimension of the design matrix
    """
    print(f"\n\nThe true category frequencies were: {true_category_probs_K}")

    for (
        category_probability_function_model_type,
        category_probability_function,
    ) in CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE.items():
        print(
            f"\n---Results below we use the betas (inferred via HMC) from {categorical_model_type} "
            f"within the {category_probability_function_model_type} category probs---\n"
        )
        for N in betas_SKM_by_N.keys():
            if np.shape(betas_SKM_by_N[N])[2] != 1:
                raise ValueError(
                    f"This report function is meant to be applied "
                    f"only in the intercepts-only case.  However, we have found that M, the "
                    f"dimensionality of the design matrix, does not equal 1"
                )
            beta_star = betas_SKM_by_N[N].mean(axis=0).T
            features = np.ones(1)[:, np.newaxis]
            probs_inferred = category_probability_function(features, beta_star)[0]
            print(f"For  N={N}, the inferred category frequences are {probs_inferred}")

        # print(f"\t (The empirical category frequencies were: {y_train__one_hot_NK[:N].mean(axis=0)}.)")


def report_on_experiment_with_covariates(
    categorical_model_type: CategoricalModelType,
    betas_SKM_by_N: Dict[int, np.array],
    y_test__one_hot_NK: np.array,
    x_test_NM: np.array,
):
    """
    Print a table giving the heldout log likelihood (i.e., the log likelihood on test set data)

    Arguments:
        categorical_model_type: the type of categorical model used when determining the regression weights (betas)
            provided in argument `betas_SKM_by_N`
        betas_SKM_by_N : dict mapping number of samples (a subset of the training data) to sampled betas;
            the sampled betas have shape (S,K), where S is the number of MCMC samples, K
            is the number of categories, and M is the dimension of the design matrix
        y_test__one_hot_NK: numpy array with shape (N,K), where N is the number of samples and
            K is the number of categories.  These are the observed data (for the test set).
        X_test_NM: numpy array with shape (N,M), where N is the number of samples and
            M is the number of covariates.  This is the design matrix (for the test set).
    """

    for (
        category_probability_function_model_type,
        category_probability_function,
    ) in CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE.items():
        print(
            f"\n---Results below we use the betas (inferred via HMC) from {categorical_model_type} "
            f"within the {category_probability_function_model_type} category probs---\n"
        )
        for N in betas_SKM_by_N.keys():
            beta_MK = betas_SKM_by_N[N].mean(axis=0).T
            category_probs = category_probability_function(x_test_NM, beta_MK)
            choice_probs = category_probs[np.where(y_test__one_hot_NK)]
            log_like = np.sum(np.log(choice_probs))
            print(f"For  N={N}, the heldout log likelihood is {log_like}")
