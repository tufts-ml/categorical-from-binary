"""
Let us easily toggle between logit and probit, at least in the case of a normal
prior on the regression weights  
"""

from enum import Enum
from typing import Optional, Union

import numpy as np
from scipy.sparse import spmatrix

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.ib_cavi.multi.ib_logit.inference import (
    compute_multiclass_logit_vi_with_polya_gamma_augmentation,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.multi.structs import (
    CAVI_Results,
    VariationalParams,
)
from categorical_from_binary.types import NumpyArray2D


class IB_Model(Enum):
    LOGIT = 1
    PROBIT = 2


def compute_ib_cavi_with_normal_prior(
    ib_model: IB_Model,
    labels: Union[NumpyArray2D, spmatrix],
    covariates: Union[NumpyArray2D, spmatrix],
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_mean_elbo: float = -np.inf,
    use_autoregressive_design_matrix: bool = False,
    labels_test: Optional[NumpyArray2D] = None,
    covariates_test: Optional[NumpyArray2D] = None,
    prior_beta_mean: Optional[NumpyArray2D] = None,
    prior_beta_precision: Optional[NumpyArray2D] = None,
    variational_params_init: Optional[VariationalParams] = None,
    verbose: bool = True,
) -> CAVI_Results:
    if ib_model is IB_Model.PROBIT:
        inference_function = compute_multiclass_probit_vi_with_normal_prior
    elif ib_model is IB_Model.LOGIT:
        inference_function = compute_multiclass_logit_vi_with_polya_gamma_augmentation
    else:
        raise ValueError(f"I don't understand the model type {ib_model}")
    return inference_function(
        labels,
        covariates,
        max_n_iterations,
        convergence_criterion_drop_in_mean_elbo,
        use_autoregressive_design_matrix,
        labels_test,
        covariates_test,
        prior_beta_mean,
        prior_beta_precision,
        variational_params_init,
        verbose,
    )


def cbm_link_from_ib_model(ib_model: IB_Model):
    if ib_model == IB_Model.LOGIT:
        return Link.CBM_LOGIT
    elif ib_model == IB_Model.PROBIT:
        return Link.CBM_PROBIT
    else:
        raise ValueError


def cbc_link_from_ib_model(ib_model: IB_Model):
    if ib_model == IB_Model.LOGIT:
        return Link.CBC_LOGIT
    elif ib_model == IB_Model.PROBIT:
        return Link.CBC_PROBIT
    else:
        raise ValueError
