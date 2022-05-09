"""
Purpose:
    Some helper functions for running experiments on CBC-Probit type models with HMC in the special case where
    everything is intercepts only.

Details:
    We want to infer betas, using HMC, for a variety of models (IB, CBC, CBM) and category probability formulas
    (CBC, SO).  We can study different samples sizes N , number of categories K,
    and true category probabilities p .. 

Reference: 
    https://github.com/tufts-ml/CBC-models/blob/main/experiments/CBCProbitDemo_K2.ipynb


Naming Conventions:
    I often use MCH's naming convention, which is to give the dimensionality of each array
    after the last underscore
"""

from enum import Enum
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from scipy.sparse import isspmatrix


jax.config.update("jax_enable_x64", True)

# Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.
#   Reference: https://github.com/pyro-ppl/pyro
# NumPyro is a small probabilistic programming library that provides a NumPy backend for Pyro.
#   Reference: http://num.pyro.ai/en/stable/getting_started.html

import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.handlers
import numpyro.infer
from numpyro.infer import init_to_value


class CategoricalModelType(int, Enum):
    """
    NOTE:
        Inherting from `int` as well as `Enum` makes this json serializable
    """

    CBC_PROBIT = 1
    CBM_PROBIT = 2
    IB_PROBIT = 3
    CBC_LOGIT = 4
    CBM_LOGIT = 5
    IB_LOGIT = 6
    SOFTMAX = 7  # softmax can also be considered non-identified multi-logit.


def create_categorical_model(
    N: int,
    categorical_model_type: CategoricalModelType,
    prior_mean: float,
    prior_stddev: float,
    y_one_hot_NK: Optional[np.array] = None,
    x_NM: Optional[np.array] = None,
    add_bias_to_design_matrix: bool = False,
):
    """
    Arguments:
        N: number of samples
        y_one_hot_NK: numpy array with shape (N,K), where N is the number of samples and
            K is the number of categories.  These are the observed data.
        X_NM: numpy array with shape (N,M), where N is the number of samples and
            M is the number of covariates.  This is the design matrix.
        add_bias_to_design_matrix: bool.  If true, a vector of all ones is appended to
            the design matrix
        prior_mean: prior mean on the regression weights for each category
        prior_stddev: prior std on the regression weights for each category
    """
    K = np.shape(y_one_hot_NK)[1]

    if x_NM is None:
        x_NM = np.ones((N, 1))
    elif x_NM is not None and add_bias_to_design_matrix:
        x_NM = np.append(x_NM, np.ones(N, 1), 1)
    M = np.shape(x_NM)[1]

    # Using MCH's naming convention here, which is to give the dimensionality of each array
    # after the last underscore
    beta_KM = numpyro.sample(
        "beta_KM", dist.Normal(prior_mean, prior_stddev).expand([K, M])
    )
    utility_NK = jnp.dot(x_NM[:N], beta_KM.T)
    if categorical_model_type == CategoricalModelType.IB_PROBIT:
        y_NK = numpyro.sample(
            "y_NK",
            dist.Bernoulli(probs=jstats.norm.cdf(utility_NK)).expand([N, K]),
            obs=y_one_hot_NK[:N],
        )
        return
    elif categorical_model_type == CategoricalModelType.IB_LOGIT:
        y_NK = numpyro.sample(
            "y_NK",
            dist.Bernoulli(probs=jstats.logistic.cdf(utility_NK)).expand([N, K]),
            obs=y_one_hot_NK[:N],
        )
        return
    else:
        if categorical_model_type == CategoricalModelType.CBC_PROBIT:
            p_unnormalized_NK = jstats.norm.cdf(utility_NK) / jstats.norm.cdf(
                -utility_NK
            )
        elif categorical_model_type == CategoricalModelType.CBM_PROBIT:
            p_unnormalized_NK = jstats.norm.cdf(utility_NK)
        elif categorical_model_type == CategoricalModelType.CBC_LOGIT:
            p_unnormalized_NK = jstats.logistic.cdf(utility_NK) / jstats.logistic.cdf(
                -utility_NK
            )
        elif categorical_model_type == CategoricalModelType.CBM_LOGIT:
            p_unnormalized_NK = jstats.logistic.cdf(utility_NK)
        elif categorical_model_type == CategoricalModelType.SOFTMAX:
            p_unnormalized_NK = jnp.exp(utility_NK)

        p_NK = (p_unnormalized_NK.T / jnp.sum(p_unnormalized_NK, 1)).T
        y_NK = numpyro.sample(  # noqa
            "y_NK",
            dist.Categorical(probs=p_NK).expand([N]),
            obs=y_one_hot_NK[:N].argmax(axis=1),
        )


def run_nuts_on_categorical_data(
    num_warmup: int,
    num_samples: int,
    Nseen_list: List[int],
    create_categorical_model: Callable,
    categorical_model_type: CategoricalModelType,
    y_train__one_hot_NK: np.array,
    x_train_NM: Optional[np.array] = None,
    prior_mean: float = 0.0,
    prior_stddev: float = 1.0,
    random_seed: int = 0,
) -> Dict[int, np.array]:
    """
    Run `NUTS` (Hoffman and Gelman's No-U Turn Sampler) on categorical data.

    Arguments:
        Nseen_list: a List of integers. Each element in the list designates a subsample
            size for running NUTS. (The idea is that, downstream, we can investigate the results
            of inference for various sample sizes.)
        y_train__one_hot_NK: an np.array with shape (N,K), where N is a number of samples and K is the
            number of categories.  position (n,k) = 1 if the n-th sample used the k-th category, and is 0 otherwise.
            Thus, summing across columns produces a (N,)-shaped vector of all 1's.
        X_train_NM: numpy array with shape (N,M), where N is the number of samples and
            M is the number of covariates.   If None, the model will automatically populate it
            with a (N,1) vector of all 1's in order to run an intercepts-only model

    Returns:
        dict mapping number of samples (a subset of the training data) to sampled betas;
        the sampled betas have shape (S,K,M), where S is the number of MCMC samples, K
        is the number of categories, and M is the dimensionality of the design matrix

    Notes:
        The initialization strategy is to initialize the regression weight matrix beta to all 0's.
    """

    betas_SKM_by_N = dict()

    # convert to dense if sparse
    if isspmatrix(x_train_NM):
        x_train_NM = np.array(x_train_NM.todense())
    if isspmatrix(y_train__one_hot_NK):
        y_train__one_hot_NK = np.array(y_train__one_hot_NK.todense(), dtype=int)

    for _, Nseen in enumerate(Nseen_list):

        if x_train_NM is None:
            x_train_NM = np.ones((Nseen, 1))

        # JAX uses a modern Threefry counter-based PRNG that’s splittable.
        # Reference: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
        # TODO: Get a better understanding of this design.
        rng_key = jax.random.PRNGKey(random_seed)
        rng_key, rng_key_ = jax.random.split(rng_key)
        n_covariates, n_categories = (
            np.shape(x_train_NM)[1],
            np.shape(y_train__one_hot_NK)[1],
        )
        model_for_nuts = numpyro.infer.NUTS(
            create_categorical_model,
            init_strategy=init_to_value(
                values={"beta_KM": jnp.zeros((n_categories, n_covariates))}
            ),
        )
        sampler = numpyro.infer.MCMC(
            model_for_nuts, num_warmup=num_warmup, num_samples=num_samples
        )

        sampler.run(
            rng_key_,
            N=Nseen,
            categorical_model_type=categorical_model_type,
            prior_mean=prior_mean,
            prior_stddev=prior_stddev,
            y_one_hot_NK=y_train__one_hot_NK,
            x_NM=x_train_NM,
        )

        # S is the number of samples
        # TODO: This function doesn't know anything about a `beta_KM`!!!  So the current factoring is not
        # appropriate.
        beta_SKM = sampler.get_samples()["beta_KM"]
        betas_SKM_by_N[Nseen] = beta_SKM

    return betas_SKM_by_N
