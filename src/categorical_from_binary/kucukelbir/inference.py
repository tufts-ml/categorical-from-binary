"""
We directly implement ADVI for cat models using Kucukelbir algo 1 

Reference to autograd:
https://jax.readthedocs.io/en/latest/notebooks/quickstart.html

Perhaps see also: 
https://luiarthur.github.io/statorial/varinf/linregpy/
"""
import collections
import sys
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import grad
from jaxlib.xla_extension import DeviceArray
from pandas.core.frame import DataFrame
from scipy.sparse import isspmatrix

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.performance_over_time.results import update_performance_results
from categorical_from_binary.types import JaxNumpyArray2D, NumpyArray1D, NumpyArray2D


###
# Metadata (Optional argument to be passed to loglike and/or logprior)
###
@dataclass
class Metadata:
    N: int  # num observations
    M: int  # num covariates
    K: int  # num predictors
    include_intercept: bool


###
# JAX versions of category probability constructions
###

# TODO:  Align this with what's in bayes_multiclass_reg.
# Perhaps make a switch that allows one to choose between np.array and jnp.array


# Each value in the `CATEGORY_PROBABILITY_FUNCTION_BY_LINK` dictionary is a function
# whose arguments are features : NumpyArray2D, betas :NumpyArray2D


class Link2(int, Enum):
    """
    Eventually want to combine this with Link
    in data generation subpackage. The naming has evolved
    overtime and I don't want to refactor the entire package right now,
    because I'm trying to meet the ICML rebuttal period deadline.

    NOTE:
        Inherting from `int` as well as `Enum` makes this json serializable
    """

    # WARNING: I have doubts that `MNP` works correctly in this ADVI setting.
    # See warnings in correspoding function for computing category probabilities
    CBC_PROBIT = 1
    SOFTMAX = 2
    MNP = 3


# WARNING: I have doubts that `MNP` works correctly in this ADVI setting.
# See warnings in correspoding function for computing category probabilities
LINK_FROM_LINK_2_NAME = {
    Link2.CBC_PROBIT.name: Link.CBC_PROBIT,
    Link2.SOFTMAX.name: Link.MULTI_LOGIT_NON_IDENTIFIED,
    Link2.MNP.name: Link.MULTI_PROBIT,
}


def compute_CBC_probit_probabilities_using_jax(
    features: JaxNumpyArray2D, beta: JaxNumpyArray2D, avoid_exact_zeros: bool = True
) -> JaxNumpyArray2D:
    """
    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Returns:
        jnp.array of shape (n_samples, n_categories)
    """
    log_cdf = jax.scipy.stats.norm.logcdf

    utilities = features @ beta  # N times K
    # # TODO: support sparse computation here
    # if scipy.sparse.issparse(utilities):
    #     utilities = utilities.toarray()

    # What this is doing is can perhaps be better understood as follows:
    #
    #  1. Construction of potentials using the CBC-Probit category probs directly from Johndrow et al (2013)
    #  from categorical_from_binary.data_generation.util import  prod_of_columns_except
    #
    #   n_obs = np.shape(features)[0]
    #   n_categories = np.shape(beta)[1]
    #   cdf = scipy.stats.norm.cdf
    #   cdf_neg_utilities = cdf(-utilities)
    #   potentials = np.zeros((n_obs, n_categories))
    #   for k in range(n_categories):
    #         potentials[:, k] = (1 - cdf_neg_utilities[:, k]) * prod_of_columns_except(
    #         cdf_neg_utilities, k
    #       )
    #
    # 2. Alternate construction of potentials, using the expression from the categorical models notes
    #      potentials=cdf(utilities)/cdf(-utilities)
    #
    # The version below though is more numerically stable
    BOUND = 1e20
    potentials = jnp.exp(log_cdf(utilities) - log_cdf(-utilities))
    potentials = jnp.clip(potentials, a_min=-BOUND, a_max=BOUND)
    normalizing_constants = jnp.sum(potentials, 1)
    cat_probs = potentials / normalizing_constants[:, jnp.newaxis]
    if avoid_exact_zeros:
        EPSILON = 1e-20
        cat_probs += EPSILON
        cat_probs /= jnp.sum(cat_probs, 1)[:, jnp.newaxis]
    return cat_probs


def compute_linear_predictors_preventing_downstream_overflow(
    features: NumpyArray2D,
    beta: NumpyArray2D,
) -> NumpyArray2D:
    """
    We want to be sure to prevent downstream overflow, since construction of category
    probabilities can make these exponentially large - consider the multi-logit link,
    which constructs category probabilities by exponentiating the linear predictors
    """
    BOUND_ON_LINEAR_PREDICTOR = 500  # to prevent overflow; value chosen somewhat arbitrarily, but I know 828 is too large
    eta = features @ beta
    return jnp.clip(
        eta, a_min=-BOUND_ON_LINEAR_PREDICTOR, a_max=BOUND_ON_LINEAR_PREDICTOR
    )


def construct_softmax_probabilities_via_jax(
    features: JaxNumpyArray2D,
    beta: JaxNumpyArray2D,
) -> JaxNumpyArray2D:
    """
    Note: This computation is not the identifiable one used during simulation.

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    M, K = np.shape(beta)
    # etas = features @ beta  # N times K
    etas = compute_linear_predictors_preventing_downstream_overflow(features, beta)
    Z = jnp.sum(jnp.array([jnp.exp(etas[:, k]) for k in range(K)]), 0)
    return jnp.array([jnp.exp(etas[:, k]) / Z for k in range(K)]).T


def construct_multi_probit_probabilities(
    features: JaxNumpyArray2D,
    beta: JaxNumpyArray2D,
    n_simulations: int = 100000,
    assume_that_random_components_of_utilities_are_independent_across_categories: bool = True,
) -> JaxNumpyArray2D:
    """
    We use a simulated probability method via Lerman and Manski (1981). It is
    summarized very nicely on pp. 120 of Kenneth Train’s book,
    Discrete Choice Models with Simualtions:

    We assume the error components are drawn from a multivariate normal N(O,I)

    Arguments:
        features: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        beta: regression weights with shape (n_features+1, n_categories-1)

    Returns:
        np.array of shape (n_samples, n_categories)
    """
    warnings.warn(
        f"I am not sure if this implementation is good.  It will certainly be slow, "
        " due to the large number of samples required to get a good approximation.  But it also "
        " requires backpropagating gradients through an argmax (and discrete variables).  I'm not "
        " sure if AD, or the larger theory of ADVI, works appropriately in this setting."
    )

    # NOTE: For now I am generating a new rng_key here inside the function each time it is called.
    # This means that we are pre-selecting the random numbers. BUT it also means I don't need
    # to change the signature of compute_log_like to take rng_keys as inputs and outputs,
    # which is annoying.  I could fix this back up laer.
    random_seed = 0
    rng_key_for_this_function_only = jax.random.PRNGKey(random_seed)

    if not assume_that_random_components_of_utilities_are_independent_across_categories:
        # TODO: Implement this.  Unlike softmax/multi-logit, multi-probit has additional
        # flexibility due to the correlated latent factors, which allows for departure
        # from the IIA assumption.  We might want to investigate behavior in this regime.
        raise NotImplementedError

    N = np.shape(features)[0]
    K = np.shape(beta)[1]
    etas = compute_linear_predictors_preventing_downstream_overflow(features, beta)

    choices_per_observation_over_simulations = jnp.zeros((N, K), dtype=int)
    for s in range(n_simulations):
        # the below needs to be called every time we want new random numbers
        (
            rng_key_for_this_function_only,
            rng_subkey_for_this_function_only,
        ) = jax.random.split(rng_key_for_this_function_only)

        # Here is where we assume that the random components are uncorrelated.
        random_contributions = jax.random.normal(
            rng_key_for_this_function_only, shape=(1, K)
        )
        random_utilities = etas + random_contributions  # (n_samples x n_categories)
        choices = jnp.argmax(random_utilities, 1)  # (n_sample,)
        simulated_labels = jax.nn.one_hot(choices, K)
        choices_per_observation_over_simulations += simulated_labels

    category_probs = choices_per_observation_over_simulations / n_simulations
    return category_probs


JAX_CATEGORY_PROBABILITY_FUNCTION_BY_LINK2 = {
    Link2.CBC_PROBIT: compute_CBC_probit_probabilities_using_jax,
    Link2.SOFTMAX: construct_softmax_probabilities_via_jax,
    Link2.MNP: construct_multi_probit_probabilities,
}

### ### ### ###
### Likelihood function
### ### ### ###


# helper function from advi repo rewritten to use jnp.arrays instead of torch.tensors


def get_P(M: int, K: int, include_intercept: bool = True):
    """
    P is the total number of parameters
    """
    if include_intercept:
        M_prime = M + 1
    else:
        M_prime = M
    return M_prime * K


def beta_matrix_from_vector(
    beta_vector: jnp.array, M: int, K: int, include_intercept: bool = True
) -> jnp.array:
    """
    Arguments:
        M: num covariates.  We assume that the design matrix has M+1 entries per category
            due to the inclusion of an intercept.
        K: num categories
    """
    if include_intercept:
        M_prime = M + 1
    else:
        M_prime = M
    return beta_vector.reshape((M_prime, K))


def standardize(z: jnp.array, mus: jnp.array, sigmas: jnp.array):
    return (z - mus) / sigmas


def unstandardize(eps: jnp.array, mus: jnp.array, sigmas: jnp.array):
    return sigmas * eps + mus


def sample_from_standard_gaussian_P_times_and_return_new_rng_key(
    rng_key: DeviceArray, P: int
):
    # the below needs to be called every time we want new random numbers
    rng_key, rng_subkey = jax.random.split(rng_key)
    return jax.random.normal(rng_subkey, (1, P)), rng_key


def compute_log_like(
    beta_flattened: jnp.array,
    data: Tuple[jnp.array, jnp.array],
    full_train_size: int,
    metadata,
    link2: Link2,
):
    # TODO: full_train_size can now be obtained from metadata!
    X, y = data[0], data[1]  # unpack tuple
    beta_matrix = beta_matrix_from_vector(
        beta_flattened, metadata.M, metadata.K, metadata.include_intercept
    )
    # cat_probs = compute_CBC_probabilities_using_jax(X, beta_matrix)
    function_to_make_cat_probs = JAX_CATEGORY_PROBABILITY_FUNCTION_BY_LINK2[link2]
    cat_probs = function_to_make_cat_probs(X, beta_matrix)
    N_batch = len(y)
    unit_likelihoods_for_this_batch = cat_probs[jnp.arange(N_batch), y]
    return jnp.log(unit_likelihoods_for_this_batch).sum() * full_train_size / N_batch


# # gives derivative w.r.t. first argument of log_like (beta_flattened)
# derivative_log_like = grad(log_like)


def compute_log_prior(
    beta_flattened: jnp.array,
):
    return jax.scipy.stats.norm.logpdf(beta_flattened, loc=0, scale=1).sum()


def compute_log_joint(
    beta_flattened: jnp.array,
    data: Tuple[jnp.array, jnp.array],
    full_data_size: int,
    metadata,
    link2: Link2,
):
    ll = compute_log_like(beta_flattened, data, full_data_size, metadata, link2)
    lp = compute_log_prior(beta_flattened)
    return ll + lp


# gives derivative w.r.t. first argument of log_joint (beta_flattened)
compute_grad_log_joint = grad(compute_log_joint)


def compute_grad_omegas_from_grad_log_joint(
    grad_log_joint: jnp.array,
    eps: jnp.array,
    omegas: jnp.array,
):
    """
    Arguments:
        grad_log_joint: jnp.array with shape (1,P) (where P is the number of model parameters)
            that is the return value of `compute_grad_log_joint`
        eps: jnp.array with shape (1,P) that is a sample from iid univariate N(0,1)s.
        omegas: jnp.array with shape (P,) which gives the log variational standard deviations.
    """
    # TODO: Is there any reason to have some arrays have shape (P,) and others (1,P)?
    # This was probably an acccident.

    diag_entries_of_gradient_of_inverse_standardization_function = eps * jnp.exp(omegas)
    # entries of the diagonal matrix against which we want to multiply the grad_log_joint.
    # we can just use broadcasting for this

    return (
        grad_log_joint * diag_entries_of_gradient_of_inverse_standardization_function
        + 1
    )


###
# STEP SIZES
###


def initialize_emas_and_return_rng_key(
    rng_key: DeviceArray,
    mus: jnp.array,
    omegas: jnp.array,
    data_train,
    full_train_size,
    metadata,
    link2: Link2,
):
    P = get_P(metadata.M, metadata.K, metadata.include_intercept)
    eps, rng_key = sample_from_standard_gaussian_P_times_and_return_new_rng_key(
        rng_key, P
    )
    beta_flattened = unstandardize(eps, mus, jnp.exp(omegas))

    grad_log_joint = compute_grad_log_joint(
        beta_flattened,
        data_train,
        full_train_size,
        metadata,
        link2,
    )

    #  these formulas for gradients work because we're taking one MC sample; see document.
    grad_mus = grad_log_joint
    grad_omegas = compute_grad_omegas_from_grad_log_joint(grad_log_joint, eps, omegas)

    ema_mus_init = grad_mus**2
    ema_omegas_init = grad_omegas**2
    return ema_mus_init, ema_omegas_init, rng_key


def update_ewa_of_squared_gradients(ewa_prev, grad_squared, alpha=0.1):
    return alpha * grad_squared + (1.0 - alpha) * ewa_prev


def compute_step_size(ewa, it, lr, tau=1.0, EPSILON=10 ** (-16)):
    return lr * it ** (-0.5 + EPSILON) / (tau + jnp.sqrt(ewa))


###
# INFERENCE
###


def do_advi_inference_via_kucukelbir_algo(
    labels_train: NumpyArray1D,
    covariates_train: NumpyArray2D,
    metadata: Metadata,
    link2: Link2,
    n_advi_iterations: int,
    lr: float,
    random_seed: int,
    alpha: float = 0.1,
    EPSILON: float = 10 ** (-16),
    tau: float = 1.0,
    labels_test: Optional[NumpyArray2D] = None,
    covariates_test: Optional[NumpyArray2D] = None,
    eval_every: int = 1,
) -> Tuple[NumpyArray2D, NumpyArray2D]:
    """
    Arguments:
        eval_every : int
            Evaluate performance every `eval_every`th iteration

    """
    # set random number generator
    rng_key = jax.random.PRNGKey(random_seed)

    # convert to dense if sparse
    if isspmatrix(covariates_train):
        covariates_train = np.array(covariates_train.todense())
    if isspmatrix(covariates_test):
        covariates_test = np.array(covariates_test.todense())
    if isspmatrix(labels_train):
        labels_train = np.array(labels_train.todense(), dtype=int)
    if isspmatrix(labels_test):
        labels_test = np.array(labels_test.todense(), dtype=int)

    # convert dataset to jnp arrays
    X_train = jnp.array(covariates_train)
    y_train = jnp.array(np.argmax(labels_train, 1))
    data_train = (X_train, y_train)

    # construct some meta info
    M, K, include_intercept = (
        metadata.M,
        metadata.K,
        metadata.include_intercept,
    )
    P = get_P(M, K, include_intercept)
    full_train_size = len(data_train[0])

    # initialize variational parameters
    mus = jnp.zeros(P)
    omegas = jnp.zeros(P)  # log standard deviations

    # initialize ema for variational parameters
    ema_mus, ema_omegas, rng_key = initialize_emas_and_return_rng_key(
        rng_key, mus, omegas, data_train, full_train_size, metadata, link2
    )

    # Prepare performance results
    performance_over_time_as_dict = collections.defaultdict(list)
    elapsed_time_for_advi_iterations = 0.0
    beta_mean = np.array(beta_matrix_from_vector(mus, M, K, include_intercept))
    link = LINK_FROM_LINK_2_NAME[link2.name]
    update_performance_results(
        performance_over_time_as_dict,
        covariates_train,
        labels_train,
        beta_mean,
        elapsed_time_for_advi_iterations,
        link=link,
        covariates_test=covariates_test,
        labels_test=labels_test,
    )

    # Highlight initial log like for reference
    mean_log_like_init = performance_over_time_as_dict[
        f"train mean log likelihood with {link.name}"
    ][-1]
    print(f"Init training mean log like  : {mean_log_like_init :.02f}")

    for it in jnp.arange(1, n_advi_iterations + 1):

        start_time_for_this_advi_iteration = time.time()

        mean_train_log_like_curr = performance_over_time_as_dict[
            f"train mean log likelihood with {link.name}"
        ][-1]
        mean_test_log_like_curr = performance_over_time_as_dict[
            f"test mean log likelihood with {link.name}"
        ][-1]
        END_OF_PRINT_STATEMENT = "\n"
        # "\r" is better if working locally, but won't show up in logs in cluster
        print(
            f"Iteration: {it}/{n_advi_iterations}. Last train mean log like: {mean_train_log_like_curr: .02f}.  Last test mean log like: {mean_test_log_like_curr: .02f}",
            end=END_OF_PRINT_STATEMENT,
        )
        sys.stdout.flush()

        if jnp.isnan(mean_train_log_like_curr):
            break

        # make beta_flattened by sampling from MVN with P dimensions, and then run it through standardization function,
        # which depends on variational parameters (m,s) = (m, log omega)
        eps, rng_key = sample_from_standard_gaussian_P_times_and_return_new_rng_key(
            rng_key, P
        )
        beta_flattened = unstandardize(eps, mus, jnp.exp(omegas))

        grad_log_joint = compute_grad_log_joint(
            beta_flattened,
            data_train,
            full_train_size,
            metadata,
            link2,
        )

        # if jnp.isnan(grad_log_joint).any():
        #     print("A nan has appeared in the gradient of the log joint!")
        #     breakpoint()

        # update mus
        grad_mus = grad_log_joint  # since we're taking one MC sample; see document.
        ema_mus = update_ewa_of_squared_gradients(ema_mus, grad_mus**2, alpha=alpha)
        rho_mus = compute_step_size(ema_mus, it, lr, tau=tau, EPSILON=EPSILON)
        mus += rho_mus * grad_mus

        # update omegas
        grad_omegas = compute_grad_omegas_from_grad_log_joint(
            grad_log_joint, eps, omegas
        )
        ema_omegas = update_ewa_of_squared_gradients(
            ema_omegas, grad_omegas**2, alpha=alpha
        )
        rho_omegas = compute_step_size(ema_omegas, it, lr, tau=tau, EPSILON=EPSILON)
        omegas += rho_omegas * grad_omegas

        end_time_for_this_advi_iteration = time.time()
        elapsed_time_for_advi_iterations += (
            end_time_for_this_advi_iteration - start_time_for_this_advi_iteration
        )

        if it % eval_every == 0:
            beta_mean = np.array(beta_matrix_from_vector(mus, M, K, include_intercept))
            update_performance_results(
                performance_over_time_as_dict,
                covariates_train,
                labels_train,
                beta_mean,
                elapsed_time_for_advi_iterations,
                link=LINK_FROM_LINK_2_NAME[link2.name],
                covariates_test=covariates_test,
                labels_test=labels_test,
            )

    beta_mean = beta_matrix_from_vector(mus, M, K, include_intercept)
    beta_stds = beta_matrix_from_vector(jnp.exp(omegas), M, K, include_intercept)
    holdout_performance_over_time = pd.DataFrame(performance_over_time_as_dict)
    return np.array(beta_mean), np.array(beta_stds), holdout_performance_over_time


###
# RESULTS CLASS
###1
@dataclass
class ADVI_Results:
    beta_mean_ADVI: NumpyArray2D
    beta_std_ADVI: NumpyArray2D
    performance_ADVI: DataFrame
