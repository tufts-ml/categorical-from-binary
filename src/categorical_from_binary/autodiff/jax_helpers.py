from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize
from jax.scipy.stats.norm import cdf as jcdf


jnp.set_printoptions(precision=3, suppress=True)


def compute_softmax_predictions(beta: jnp.array, X: jnp.array) -> jnp.array:
    """
    Note: This computation is not the identifiable one used during simulation.

    Arguments:
        beta: MxK
        X: NxM
    Returns:
        probs: NxK
    """
    M, K = jnp.shape(beta)
    etas = X @ beta  # N times K

    Z = jnp.sum(jnp.array([jnp.exp(etas[:, k]) for k in range(K)]), 0)
    return jnp.array([jnp.exp(etas[:, k]) / Z for k in range(K)]).T


def compute_CBM_Probit_predictions(
    beta: jnp.array, X: jnp.array, return_numpy: bool = False
) -> jnp.array:
    """
    Arguments:
        beta: MxK
        X: NxM
    Returns:
        probs: NxK
    """
    M, K = jnp.shape(beta)
    etas = X @ beta  # N times K

    Z = jnp.sum(jnp.array([jcdf(etas[:, k]) for k in range(K)]), 0)
    CBM_predictions = jnp.array([jcdf(etas[:, k]) / Z for k in range(K)]).T
    if return_numpy:
        return np.copy(CBM_predictions)
    else:
        return CBM_predictions


def compute_CBC_Probit_predictions(
    beta: jnp.array, X: jnp.array, return_numpy: bool = False
) -> jnp.array:
    """
    Arguments:
        beta: MxK
        X: NxM
    Returns:
        probs: NxK
    """
    EPSILON = 1e-10

    M, K = jnp.shape(beta)
    etas = X @ beta  # N times K
    # Add epsilon to avoid psi taking the form of 1/0.
    psis = jcdf(etas) / (jcdf(-etas) + EPSILON)
    Z = jnp.sum(psis, 1)
    CBC_predictions = (psis.T / Z).T
    if return_numpy:
        return np.copy(CBC_predictions)
    else:
        return CBC_predictions


def compute_IB_Probit_predictions(beta: jnp.array, X: jnp.array) -> jnp.array:
    """
    Arguments:
        beta: MxK
        X: NxM
    Returns:
        probs: NxK
    """
    M, K = jnp.shape(beta)
    etas = X @ beta  # N times K

    return jnp.array([jcdf(etas[:, k]) for k in range(K)]).T


def compute_training_loss(
    beta, X, y, category_probability_function: Callable, is_truly_categorical: bool
) -> float:
    """
    Arguments:
        beta: MxK
        X: NxM
        y: NxK (one-hot encoded)
        category_probability_function: Callable
            arguments are beta and X, returns NxK category probabilities
        is_truly_categorical : bool
            this is fully determined by the category probability function
            this determines how the training loss is computed
    """
    preds = category_probability_function(beta, X)
    if is_truly_categorical:
        return -jnp.mean(jnp.log(preds[y == 1]))
    else:
        binary_probs = preds * y + (1 - preds) * (1 - y)
        return -jnp.mean(jnp.log(binary_probs))


def compute_training_loss_with_beta_flattened(
    beta_flattened,
    M: int,
    K: int,
    X,
    y,
    category_probability_function: Callable,
    is_truly_categorical: bool,
) -> float:
    """
    Arguments:
        beta_flattened: (MxK, 1)
            This can be taken to be the result of taking (M,K) beta array called `beta` upon which is performed
            jnp.ravel(beta). This produces a (MxK, 1) beta array which is flattened by rows (i.e. the entire first
            row is followed by the entire second row, and so forth).  We use beta_flattened instead of the beta matrix
            because jax's scipy.optimize requires the ijnput to be a 1D array
        M: first dimension for beta once we "unflatten" beta_flattened
        K: first dimension for beta once we "unflatten" beta_flattened
        X: NxM
        y: NxK (one-hot encoded)
        category_probability_function: Callable
            arguments are beta and X, returns NxK category probabilities
        is_truly_categorical : bool
            this is fully determined by the category probability function
            this determines how the training loss is computed
    Returns:
        probs: NxK
    """
    beta = jnp.reshape(beta_flattened, (M, K))
    return compute_training_loss(
        beta, X, y, category_probability_function, is_truly_categorical
    )


def optimize_beta(
    beta: jnp.array,
    X: jnp.array,
    y: jnp.array,
    category_probability_function: Callable,
    is_truly_categorical: bool,
    verbose: bool = True,
) -> np.array:
    """
    Arguments:
        beta: MxK
        X: NxM
        y: NxK (one-hot encoded)
        category_probability_function: Callable
            arguments are beta and X, returns NxK category probabilities
    """
    if verbose:
        print(
            f"Inital training loss:{compute_training_loss(beta, X, y, category_probability_function, is_truly_categorical)} "
        )
        print(
            f"Initial predictions (first five samples):{category_probability_function(beta,X)[:5]}"
        )

    M, K = jnp.shape(beta)
    beta_flattened = jnp.ravel(beta)

    result = minimize(
        compute_training_loss_with_beta_flattened,
        beta_flattened,
        args=(M, K, X, y, category_probability_function, is_truly_categorical),
        method="BFGS",
    )
    beta_star_flattened = result.x
    beta_star = jnp.reshape(beta_star_flattened, (M, K))

    if verbose:
        print(
            f"New training loss: {compute_training_loss(beta_star, X, y, category_probability_function, is_truly_categorical)}"
        )
        print(
            f"New predictions (first five samples):{category_probability_function(beta_star,X)[:5]}"
        )

    return np.array(beta_star)


optimize_beta_for_IB_model = partial(
    optimize_beta,
    category_probability_function=compute_IB_Probit_predictions,
    is_truly_categorical=False,
)

optimize_beta_for_CBM_model = partial(
    optimize_beta,
    category_probability_function=compute_CBM_Probit_predictions,
    is_truly_categorical=True,
)

optimize_beta_for_CBC_model = partial(
    optimize_beta,
    category_probability_function=compute_CBC_Probit_predictions,
    is_truly_categorical=True,
)

optimize_beta_for_softmax_model = partial(
    optimize_beta,
    category_probability_function=compute_softmax_predictions,
    is_truly_categorical=True,
)
