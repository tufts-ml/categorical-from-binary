"""
Note: this module is highly redundant with binary_probit.inference.structs, but there were still some key differences,
and it seemed simpler at this point to just have a lot of overlap in the data classes than to try to share data class
code.  Perhaps this will bite me in the butt later.  I guess one questions is whether it's possible to have
parametric data classes. 
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import spmatrix

from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


class PriorType(Enum):
    NORMAL = 1
    NORMAL_GAMMA = 2


@dataclass
class VariationalBeta:
    """
    Attributes:
        mean: has shape (M,K), where M is the number of covariates and K is
            the number of categories.

            The IB-probit inference code will make this sparse
            if the covariates passed to it are sparse.
        cov: has shape (M,M) or (M,M,K). The former is referred to as the "compact"
            representation and can be used to save space when the covariance matrix is identical
            across categories (as it is when doing variational inference on a
            standard *CBC-Probit model).

            The IB-probit inference code will make this sparse
            if the covariates passed to it are sparse.
    """

    mean: Union[NumpyArray2D, spmatrix]
    cov: Union[NumpyArray2D, NumpyArray3D, spmatrix]


@dataclass
class VariationalZs:
    """
    Attributes:
        parent_mean: has shape (N,K), where N is the number of observations and K is
            the number of categories.
        parent_var: has shape (N,K), where N is the number of observations and K is
            the number of categories.   Optional because this these are set uniformly
            to 1 when doing the IB probit model.
    """

    parent_mean: NumpyArray2D
    parent_var: Optional[NumpyArray2D] = None


@dataclass
class VariationalTaus:
    """
    Attributes:
        a: has shape (M,K), where M is the number of covariates and K is the number of categories
        d: has shape (M,K), where M is the number of covariates and K is the number of categories
    """

    a: NumpyArray2D
    c: float
    d: NumpyArray2D


@dataclass
class VariationalOmegas:
    """
    Attributes:
        b: has shape (N,K), where N is the number of observations and K is
            the number of categories.
        c: has shape (N,K), where N is the number of observations and K is
            the number of categories.
    """

    b: NumpyArray2D
    c: NumpyArray2D


@dataclass
class VariationalParams:
    """
    Attributes:
        zs: Optional because it needn't be set at initialization.
        taus:  Optional because this random variable exists only for a normal-gamma
            prior (which incidentally is currently supported for the CBC-Probit model
            but not the CBC-Logit model)
        omegas: Optional because this random variable exists only for the CBC-Logit model,
            and not for the CBC-Probit model, due to polya gamma augmentation.
    """

    beta: VariationalBeta
    zs: Optional[VariationalZs] = None
    taus: Optional[VariationalTaus] = None
    omegas: Optional[VariationalOmegas] = None


@dataclass
class CAVI_Results:
    variational_params: VariationalParams
    performance_over_time: Optional[DataFrame] = None


@dataclass
class Precomputables:
    """
    Compute these once in advance, for speed:
    """

    beta_cov_init_times_design_matrix_transposed: Optional[NumpyArray2D] = None
    design_matrix_t_design_matrix: Optional[NumpyArray2D] = None
    shrinkage_grouping_strategy: Optional[NumpyArray2D] = None


@dataclass
class ELBO_Stats:
    convergence_criterion_drop_in_mean_elbo: float
    previous_mean_elbo: float = -np.inf
    mean_elbo: float = -np.inf
    drop_in_mean_elbo: float = np.inf
