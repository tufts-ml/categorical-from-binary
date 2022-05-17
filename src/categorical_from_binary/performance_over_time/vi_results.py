"""
Note: this module is highly redundant with ib_cavi.multi.structs, but we want to have some generic
structs that will allows us to save/load in a uniform manner across IB-CAVI and ADVI (so that
we can easily compare the results of inference, e.g. in the intrusion detection experiment)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from pandas.core.frame import DataFrame
from scipy.sparse import spmatrix

from categorical_from_binary.ib_cavi.multi.structs import CAVI_Results
from categorical_from_binary.kucukelbir.inference import ADVI_Results
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


class VI_Type(int, Enum):
    IB_CAVI = 1
    ADVI = 2


@dataclass
class VariationalBetaParams:
    """
    Attributes:
        mean:
            Used by: ADVI and IB-CAVI.

            Shape: (M,K), where M is the number of covariates and K is
            the number of categories.

            Sparsity conditions: The IB-probit inference code will make this sparse
            if the covariates passed to it are sparse.

        cov:
            Used by: IB-CAVI only.

            Shape: (M,M) or (M,M,K). The former is referred to as the "compact"
            representation and can be used to save space when the covariance matrix is identical
            across categories (as it is when doing variational inference on a
            standard *CBC-Probit model).

            Sparsity conditions:  The IB-probit inference code will make this sparse
            if the covariates passed to it are sparse.

        stds:
            Used by: ADVI only.

            Shape: (M,K).
    """

    mean: Union[NumpyArray2D, spmatrix]
    cov: Optional[Union[NumpyArray2D, NumpyArray3D, spmatrix]] = None
    stds: Optional[Union[NumpyArray2D, spmatrix]] = None


@dataclass
class VI_Results:
    vi_type: VI_Type
    variational_beta_params: VariationalBetaParams
    performance_over_time: Optional[DataFrame] = None


def VI_results_from_CAVI_results(CAVI_results: CAVI_Results):
    variational_beta_params = VariationalBetaParams(
        mean=CAVI_results.variational_params.beta.mean,
        cov=CAVI_results.variational_params.beta.cov,
        stds=None,
    )
    performance_over_time = CAVI_results.performance_over_time
    vi_type = VI_Type.IB_CAVI
    return (vi_type, variational_beta_params, performance_over_time)


def VI_results_from_ADVI_results(ADVI_results: ADVI_Results):
    variational_beta_params = VariationalBetaParams(
        mean=ADVI_results.beta_mean_ADVI,
        cov=None,
        stds=ADVI_results.beta_std_ADVI,
    )
    performance_over_time = ADVI_results.performance_ADVI
    vi_type = VI_Type.ADVI
    return (vi_type, variational_beta_params, performance_over_time)


# TODO: Add "write" function (see train_model in the intrustion detection experiment)
