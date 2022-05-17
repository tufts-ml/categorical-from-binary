import collections
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.ib_cavi.multi.inference import IB_Model
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


###
# Preliminaries
###


@dataclass
class BetaSamplesAndLink:
    samples: NumpyArray3D  # S x M x L
    link: Link


@dataclass
class CatProbData:
    feature_vector: NumpyArray1D
    samples: NumpyArray2D  # S x K
    link: Link


###
# Construct posterior samples (from optimization methods, IB-CAVI and ADVI, which don't yield samples)
###


def sample_beta_cavi(beta_mean, beta_cov_across_M_for_all_K, seed):
    """
    beta_mean: (M,K)
    beta_cov_across_M_for_all_K: (M,M)  (uniform across K)
    """
    np.random.seed(seed)

    beta_stds_across_M_for_all_K = np.sqrt(np.diag(beta_cov_across_M_for_all_K))
    M, K = np.shape(beta_mean)
    beta_matrix_sample = np.zeros_like(beta_mean)
    for k in range(K):
        beta_matrix_sample[:, k] = np.random.normal(
            loc=beta_mean[:, k], scale=beta_stds_across_M_for_all_K
        )
    return beta_matrix_sample


def sample_beta_advi(beta_mean, beta_stds, seed):
    """
    beta_mean: (M,L), where L in {K-1, K} depending on link function
    beta_stds: (M,L), where L in {K-1, K} depending on link function
    """
    np.random.seed(seed)

    M, L = np.shape(beta_mean)
    beta_matrix_sample = np.zeros_like(beta_mean)
    for m in range(M):
        for l in range(L):
            beta_matrix_sample[m, l] = np.random.normal(
                loc=beta_mean[m, l], scale=beta_stds[m, l]
            )
    return beta_matrix_sample


###
# Get category probability samples and variance for a FIXED covariate vector
###


def cat_prob_samples_from_beta_samples(
    feature_vector, beta_samples: NumpyArray3D, link: Link
) -> NumpyArray2D:
    """
    Arguments:
        beta_samples : array of shape (S,M,L)
            where S is the num of samples from the posterior, M is the num of covariates (incl intercept), and L
            is the number of "free" categories (i.e. those without an identifiability constraint imposed)
    """
    num_mcmc_samples, _, L = np.shape(beta_samples)
    K = np.shape(construct_category_probs(feature_vector, beta_samples[0, :, :], link))[
        1
    ]
    cat_prob_samples = np.zeros((num_mcmc_samples, K))
    for i in range(num_mcmc_samples):
        cat_prob_samples[i, :] = construct_category_probs(
            feature_vector, beta_samples[i, :, :], link
        )
    return cat_prob_samples


def construct_cat_prob_data_by_method(
    feature_vector, beta_samples_and_link_by_method: Dict[str, BetaSamplesAndLink]
) -> Dict[str, CatProbData]:
    cat_prob_data_by_method = dict()
    for method, beta_samples_and_link in beta_samples_and_link_by_method.items():
        bsl = beta_samples_and_link
        cat_prob_samples = cat_prob_samples_from_beta_samples(
            feature_vector, bsl.samples, bsl.link
        )
        cat_prob_data_by_method[method] = CatProbData(
            feature_vector, cat_prob_samples, bsl.link
        )
    return cat_prob_data_by_method


def add_bma_to_cat_prob_data_by_method(
    cat_prob_data_by_method: Dict[str, CatProbData],
    CBC_weight: float,
    ib_model: IB_Model,
) -> Dict[str, CatProbData]:
    # TODO: Make method+link an actual class so we don't have to get the
    # strings exactly right
    if ib_model == IB_Model.PROBIT:
        cbc_method = "CBC_PROBIT+IB-CAVI"
        cbm_method = "CBM_PROBIT+IB-CAVI"
        bma_method = "BMA_PROBIT+IB-CAVI"
        bma_link = Link.BMA_PROBIT
    elif ib_model == IB_Model.LOGIT:
        cbc_method = "CBC_LOGIT+IB-CAVI"
        cbm_method = "CBM_LOGIT+IB-CAVI"
        bma_method = "BMA_LOGIT+IB-CAVI"
        bma_link = Link.BMA_LOGIT
    else:
        raise ValueError("What is the ib_model?!")

    feature_vector = cat_prob_data_by_method[cbc_method].feature_vector
    samples_BMA = np.zeros_like(cat_prob_data_by_method[cbc_method].samples)
    for i in range(len(samples_BMA)):
        probs_CBC = cat_prob_data_by_method[cbc_method].samples[i]
        probs_CBM = cat_prob_data_by_method[cbm_method].samples[i]
        probs_BMA = CBC_weight * probs_CBC + (1 - CBC_weight) * probs_CBM
        samples_BMA[i] = probs_BMA
    cat_prob_data_with_bma = CatProbData(feature_vector, samples_BMA, bma_link)
    cat_prob_data_by_method[bma_method] = cat_prob_data_with_bma
    return cat_prob_data_by_method


def make_df_of_sampled_category_probs_for_each_method_and_covariate_vector(
    covariates: NumpyArray2D,
    CBC_weight: float,
    beta_samples_and_link_by_method: Dict[str, BetaSamplesAndLink],
    example_idxs: List[int],
    colors_by_methods_to_plot: Dict[str, str],
    n_categories: int,
    num_mcmc_samples: int,
    ib_model: IB_Model,
) -> DataFrame:

    # TODO: Autoinfer `n_categories` and `num_mcmc_samples`

    d = collections.defaultdict(list)

    ###
    # Make flattened dataframe of posterior category prob samples for each sample (example) in example_idx
    ###

    K = n_categories
    methods = list(colors_by_methods_to_plot.keys())

    for example_idx in example_idxs:
        print(
            f"Obtaining sampled posterior category probabilities for example index {example_idx}"
        )

        ###
        # Get posterior samples of category probs for one feature vector
        ###
        feature_vector = np.array([covariates[example_idx, :]])
        cat_prob_data_by_method = construct_cat_prob_data_by_method(
            feature_vector, beta_samples_and_link_by_method
        )

        cat_prob_data_by_method = add_bma_to_cat_prob_data_by_method(
            cat_prob_data_by_method,
            CBC_weight,
            ib_model,
        )

        for k in range(K):
            # TODO: auto infer num mcmc samples
            for i in range(num_mcmc_samples):
                for method in methods:
                    d["prob"].append(cat_prob_data_by_method[method].samples[i, k])
                    d["category"].append(k + 1)
                    d["method"].append(method)
                    d["example"].append(example_idx)
    df = pd.DataFrame(d)
    df["method"] = pd.Categorical(df.method)  # necessary?
    df["category"] = pd.Categorical(df.category)  # necessary?
    return df
