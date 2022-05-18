"""
Note: this module may be redundant with ib_cavi.multi.structs, but we want to have some generic
structs that will allows us to save/load in a uniform manner across IB-CAVI and ADVI (so that
we can easily compare the results of inference, e.g. in the intrusion detection experiment)
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import scipy
from scipy.sparse import spmatrix

from categorical_from_binary.ib_cavi.multi.structs import CAVI_Results

# from categorical_from_binary.kucukelbir.inference import ADVI_Results
from categorical_from_binary.io import ensure_dir
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


class VI_Type(int, Enum):
    IB_CAVI = 1
    ADVI = 2


@dataclass
class VI_Params:
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

    VI_type: VI_Type
    mean: Union[NumpyArray2D, spmatrix]
    cov: Optional[Union[NumpyArray2D, NumpyArray3D, spmatrix]] = None
    stds: Optional[Union[NumpyArray2D, spmatrix]] = None
    VI_hyperparams: Optional[str] = ""  # e.g. learning rate for ADVI


def VI_params_from_CAVI_results(
    CAVI_results: CAVI_Results,
):
    return VI_Params(
        VI_type=VI_Type.IB_CAVI,
        mean=CAVI_results.variational_params.beta.mean,
        cov=CAVI_results.variational_params.beta.cov,
        stds=None,
        VI_hyperparams="",
    )


# Removing due to circular import
#
# def VI_params_from_ADVI_results(
#     ADVI_results: ADVI_Results,
#     lr: float,
# ):
#     return VI_Params(
#         VI_type=VI_Type.IB_CAVI,
#         mean=ADVI_results.beta_mean_ADVI,
#         cov=None,
#         stds=ADVI_results.beta_std_ADVI,
#         VI_hyperparams=f"lr={lr}",
#     )


def VI_params_from_ADVI_means_and_stds(
    beta_mean_ADVI: NumpyArray2D,
    beta_std_ADVI: NumpyArray2D,
    lr: float,
):
    return VI_Params(
        VI_type=VI_Type.ADVI,
        mean=beta_mean_ADVI,
        cov=None,
        stds=beta_std_ADVI,
        VI_hyperparams=f"lr={lr}",
    )


def write_VI_params(
    VI_params: VI_Params,
    save_dir: str,
    time_info: str,
) -> None:
    detailed_dir = os.path.join(
        save_dir,
        "betas",
        VI_params.VI_type.name,
        VI_params.VI_hyperparams,
        time_info,
    )

    ensure_dir(detailed_dir)

    # Save beta mean
    beta_mean = VI_params.mean
    if scipy.sparse.issparse(beta_mean):
        path_to_beta_mean = os.path.join(detailed_dir, "beta_mean.npz")
        scipy.sparse.save_npz(path_to_beta_mean, beta_mean)
    else:
        path_to_beta_mean = os.path.join(detailed_dir, "beta_mean.npy")
        np.save(path_to_beta_mean, beta_mean)

    # Save beta variation information (MxM cov matrix for IB-CAVI; MxK stds for ADVI)
    beta_cov = VI_params.cov
    if beta_cov is not None:
        if scipy.sparse.issparse(beta_cov):
            path_to_beta_cov = os.path.join(detailed_dir, "beta_cov.npz")
            scipy.sparse.save_npz(path_to_beta_cov, beta_cov)
        else:
            path_to_beta_cov = os.path.join(detailed_dir, "beta_cov.npy")
            np.save(path_to_beta_cov, beta_cov)
    beta_stds = VI_params.stds
    if beta_stds is not None:
        if scipy.sparse.issparse(beta_stds):
            path_to_beta_stds = os.path.join(detailed_dir, "beta_stds.npz")
            scipy.sparse.save_npz(path_to_beta_stds, beta_stds)
        else:
            path_to_beta_stds = os.path.join(detailed_dir, "beta_stds.npy")
            np.save(path_to_beta_stds, beta_stds)


def write_VI_params_from_CAVI_results(
    CAVI_results: CAVI_Results,
    save_dir: str,
    time_info: str,
):
    VI_params = VI_params_from_CAVI_results(CAVI_results)
    write_VI_params(VI_params, save_dir, time_info)


# Removing due to circular import
#
# def write_VI_params_from_ADVI_results(
#     ADVI_results: ADVI_Results,
#     lr: float,
#     save_dir: str,
#     time_info : str,
# ):
#     VI_params = VI_params_from_ADVI_results(ADVI_results, lr)
#     write_VI_params(VI_params, save_dir, time_info)


def write_VI_params_from_ADVI_means_and_stds(
    beta_mean_ADVI: NumpyArray2D,
    beta_std_ADVI: NumpyArray2D,
    lr: float,
    save_dir: str,
    time_info: str,
):
    VI_params = VI_params_from_ADVI_means_and_stds(beta_mean_ADVI, beta_std_ADVI, lr)
    write_VI_params(VI_params, save_dir, time_info)
