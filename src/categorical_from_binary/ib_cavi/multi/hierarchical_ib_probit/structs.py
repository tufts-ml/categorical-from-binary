"""
Provides some structs (and functions) useful for passing around
data when doing coordinate ascent variational inference (CAVI)
with the hierarchical CBC-Probit regression model.
"""

from typing import List, NamedTuple

import numpy as np
from recordclass import RecordClass

from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    HierarchicalMulticlassRegressionDataset,
)
from categorical_from_binary.types import (
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    NumpyArray4D,
)


class Hyperparameters(NamedTuple):
    """
    See report for notation.  Basically:
        1. The mean (across groups) regression weights
        for response category k - denoted mu_k in the report- is taken to be N(m_0, V_0)
        2. The covariance (across groups) in regression weights
        for response category k - denoted Sigma_k in the report - is taken to be IW(nu_0, S_0)

    This class stored the hyperparameters mentioned above.
    """

    V_0: NumpyArray2D
    m_0: NumpyArray1D
    nu_0: float
    S_0: NumpyArray2D


class DataDimensions(NamedTuple):
    num_groups: int
    num_designed_features: int
    num_categories: int
    num_observations_per_group: List[int]


class VariationalParams(RecordClass):
    """
    A collection of all the variational parameters.  For reference, see the
    report in reports/ of this repo on variational inference in categorical models.

    In what follows we have:
        J: number of groups
        M: number of ``designed" features -- e.g. includes 1 for intercept, if exists
        K: number of categories
        N_j: number of observations in group j=1,..,J

    Attributes:
        beta_means: array with shape (J,M,K)
            The variational expectation on the group-specific betas.
            These are denoted tilde-mu-jk in the report.
        beta_covs:  array with shape (J,M,M,K)
            The variational covariance on the group-specific betas.
            These are denoted tilde-Sigma-jk in the report
        mu_means: array with shape (M,K)
            The variational expectation of the population-level beta mean
            These are denoted tilde-m-k in the report.
        mu_covs: array with shape (M,M,K)
            The variational covariance of the population-level beta mean
            These are denoted tilde-V-k in the report.
        Sigma_dfs : array with shape (K)
            The variational degrees of freedom on the population-level beta covariance
            for each response category k=1,..,K
        Sigma_RSSs: array with shape (M,M,K)
            The variational RSS (Residual Sum of Squares) on the population-level beta covariance
            for each response category k=1,..,K
        zs_expected: List with J entries; each element is an array with shape (N_j, K)
            The variational expectations of the latent z values.  Note that these are NOT
            the natural parameters (i.e. the means of the parent normal distributions.)
    """

    beta_means: NumpyArray3D
    beta_covs: NumpyArray4D
    mu_means: NumpyArray2D
    mu_covs: NumpyArray3D
    Sigma_dfs: NumpyArray1D
    Sigma_RSSs: NumpyArray3D
    zs_expected: List[NumpyArray2D]


def data_dimensions_from_hierarchical_dataset(
    data: HierarchicalMulticlassRegressionDataset,
) -> DataDimensions:
    """
    Usage:
        One can get short nicknames (corresponding to those used in the report)
        from a hiararchical dataset via:

        J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)
    """
    num_groups = len(data.datasets)
    num_designed_features, num_categories = np.shape(data.datasets[0].beta)
    num_observations_per_group = [len(dataset.features) for dataset in data.datasets]
    return DataDimensions(
        num_groups, num_designed_features, num_categories, num_observations_per_group
    )
