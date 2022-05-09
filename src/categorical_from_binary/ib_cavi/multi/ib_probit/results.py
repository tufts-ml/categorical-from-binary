"""
Here we compare two estimators (CBC probit and CBM Probit) for the category probabilities 
given beta-hat when 
(a) we use the variational posterior expectation for beta-hat
(b) the dataset has covariates
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.hmc.core import CategoricalModelType


@dataclass
class ResultsOnOneSplit:
    """
    Attributes:
        seed: The random seed used to split the data into train/test data
        split_dataset : SplitDataset
        beta_samples_by_model_type:  a dictionary mapping a CategoricalModelType to an np.array.
            The np.array are regression weight betas with dimensionality SKM, where S is the number of MCMC
            samples, K is the number of categories, and M is the number of covariates (including bias).
            Used for MCMC sampling
        beta_mean_by_model_type:  np.array.
            The np.array are variational expectations for betas with dimensionality KM, where K is the number of categories,
            and M is the number of covariates (including bias).
            Used for variational inference.
        beta_cov_by_model_type: np.array.
            The np.array are are variational covariances for betas.
            Used for variational inference.
    """

    seed: int = None
    split_dataset: SplitDataset = None
    beta_samples_by_model_type: Dict[CategoricalModelType, np.array] = None
    beta_mean: np.array = None
    beta_cov: np.array = None
    link_for_generating_data: Link = None
