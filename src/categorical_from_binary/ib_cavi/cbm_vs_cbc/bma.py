"""
(Approximate) Bayesian model averaging for the IB+CBC probit and IB+CBM probit category probabilities.
"""

import numpy as np
import scipy.stats

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_category_probs,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.real_elbo import (
    approximate_true_elbo_with_samples,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    do_link_from_ib_model,
    sdo_link_from_ib_model,
)
from categorical_from_binary.ib_cavi.multi.structs import VariationalBeta
from categorical_from_binary.types import NumpyArray2D


def compute_weight_on_CBC_from_bayesian_model_averaging(
    covariates_train: NumpyArray2D,
    labels_train: NumpyArray2D,
    variational_beta: VariationalBeta,
    n_monte_carlo_samples: int,
    ib_model: IB_Model,
) -> float:
    """
    Compute weight on CBC model in an (approximate) Bayesian Model Averaging of CBC and CBM
    using the approximate posterior from IB fit with mean-field variational inference.
    Assumes the CBC and CBM are equally weighted in the prior.

    The Bayesian Model Averaging is approximate because we use ELBO_m as approximations to
    the evidence p_m(x) for models m in set(CBM, CBC).
    """
    sdo_link = sdo_link_from_ib_model(ib_model)
    do_link = do_link_from_ib_model(ib_model)

    elbo_hat_CBM = approximate_true_elbo_with_samples(
        covariates_train,
        labels_train,
        sdo_link,
        variational_beta,
        n_monte_carlo_samples,
    )
    elbo_hat_CBC = approximate_true_elbo_with_samples(
        covariates_train,
        labels_train,
        do_link,
        variational_beta,
        n_monte_carlo_samples,
    )

    # use that to derive model weight (assuming 50/50 prior on each )
    CBC_weight = np.exp(
        elbo_hat_CBC - scipy.special.logsumexp([elbo_hat_CBM, elbo_hat_CBC])
    )
    return CBC_weight


def construct_category_probabilities_from_bayesian_model_averaging(
    covariates: NumpyArray2D,
    variational_beta_point_estimate: NumpyArray2D,
    CBC_weight: float,
    ib_model: IB_Model,
) -> NumpyArray2D:

    sdo_link = sdo_link_from_ib_model(ib_model)
    do_link = do_link_from_ib_model(ib_model)
    probs_CBM = construct_category_probs(
        covariates, variational_beta_point_estimate, sdo_link
    )
    probs_CBC = construct_category_probs(
        covariates, variational_beta_point_estimate, do_link
    )
    return CBC_weight * probs_CBC + (1 - CBC_weight) * probs_CBM
