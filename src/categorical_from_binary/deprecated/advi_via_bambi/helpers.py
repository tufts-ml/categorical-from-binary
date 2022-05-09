import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import bambi as bmb
import numpy as np
import pandas as pd
import pymc3
from pymc3.variational.approximations import MeanField

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


###
# Model construction
###


def create_bambi_softmax_model(
    covariates_train: NumpyArray2D,
    labels_train: NumpyArray2D,
) -> bmb.models.Model:
    """
    Bambi is a python package (BAyesian Model-Building Interface) which can construct
    models that pymc3 can then operate on.

    Note that the bambi softmax model seems to have been identified somehow.
    The results of inference seems to be setting the beta for the 0th category
    to be equal to the zero vector.

    Arguments:
        covariates_train: an np.array of shape (n_samples, n_features+1)
            Includes a ones column for the intercept term!
        labels_train:  an np.array of shape (n_samples, n_categories);
            each row is one-hot encoded
    Reference:
        In the logistic regression case:
        #https://bambinos.github.io/bambi/main/notebooks/logistic_regression.html
    """
    ### 1. Convert training data to a data frame
    covariates_train_without_intercept = covariates_train[:, 1:]
    M = np.shape(covariates_train_without_intercept)[1]
    df_train = pd.DataFrame(
        covariates_train_without_intercept, columns=[f"cov{x}" for x in range(M)]
    )

    # now add labels
    choices_train = np.argmax(labels_train, 1)
    choices_train_as_categorical_series = pd.Series(choices_train, dtype="category")
    df_train["choices"] = choices_train_as_categorical_series

    ###  2. write a formula that applies to columns of a dataframe
    list_of_covariates_as_strings = [f"cov{x}" for x in range(M)]
    covariates_as_strings = " + ".join(list_of_covariates_as_strings)
    formula_as_string = "choices ~ " + covariates_as_strings

    ### 3. Construct the softmax model
    return bmb.Model(formula_as_string, df_train, family="categorical")


###
# Inference
###


@dataclass
class IntermediateResult:
    landmark_iteration: int
    time: float
    beta_mean_vector: NumpyArray1D


def callback_to_report_time_and_variational_approximation_at_landmark_iterations(
    approx: MeanField,
    losses,
    i: int,
) -> Optional[IntermediateResult]:
    """
    Method to operate on advi object during each iteration of training

    Arguments:
        approx: instance of a class for doing ADVI from the pymc3 library.
            See https://github.com/pymc-devs/pymc/blob/main/pymc/variational/approximations.py.
        losses: no idea, it's not explained in the documentation given by ?pymc3.callbacks.Traceback
        i: current iteration.  Seems to be one-indexed

    Returns:
       (landmark iteration, time, beta_mean_vector) at landmark iterations; else it returns None.   We return
       None at non-landmark iterations to avoid hogging up a ton of memory.
    """
    # TODO: Allow us control over what counts as landmark iterations, rather than hardcoding them.
    # TODO: Or at least have it be a percentage of the total number of iterations.

    LANDMARK_ITERATION = 500
    idx_in_zero_indexing = i - 1
    if idx_in_zero_indexing % LANDMARK_ITERATION == 0:
        beta_mean_vector = approx.mean.eval()
        return IntermediateResult(idx_in_zero_indexing, time.time(), beta_mean_vector)
    else:
        return None


def get_fitted_advi_model_and_intermediate_results(
    model: bmb.models.Model,
    method_as_string: str = "advi",
    n_its: int = 10000,
    init: str = "auto",
    n_init: int = 5000,
    n_mc_samples_for_gradient_estimation: int = 500,
    random_seed: Optional[int] = None,
    my_intermediate_result_callback: Optional[
        Callable
    ] = callback_to_report_time_and_variational_approximation_at_landmark_iterations,
) -> Tuple[MeanField, Optional[List[IntermediateResult]]]:
    """
    Arguments:
        n_its: Number of ADVI iterations.  We use the same default as used by pymc3.
        init: str
            Initialization method. Defaults to ``"auto"``. The available methods are:
            * auto: Use ``"jitter+adapt_diag"`` and if this method fails it uses ``"adapt_diag"``.
            * adapt_diag: Start with a identity mass matrix and then adapt a diagonal based on the
            variance of the tuning samples. All chains use the test value (usually the prior mean)
            as starting point.
            * jitter+adapt_diag: Same as ``"adapt_diag"``, but use test value plus a uniform jitter
            in [-1, 1] as starting point in each chain.
            * advi+adapt_diag: Run ADVI and then adapt the resulting diagonal mass matrix based on
            the sample variance of the tuning samples.
            * advi+adapt_diag_grad: Run ADVI and then adapt the resulting diagonal mass matrix based
            on the variance of the gradients during tuning. This is **experimental** and might be
            removed in a future release.
            * advi: Run ADVI to estimate posterior mean and diagonal mass matrix.
            * advi_map: Initialize ADVI with MAP and use MAP as starting point.
            * map: Use the MAP as starting point. This is strongly discouraged.
            * adapt_full: Adapt a dense mass matrix using the sample covariances. All chains use the
            test value (usually the prior mean) as starting point.
            * jitter+adapt_full: Same as ``"adapt_full"``, but use test value plus a uniform jitter
            in [-1, 1] as starting point in each chain.
        n_init :  Number of initialization iterations. Only works for ``"advi"`` init methods.
            We use the same default as used by pymc3.
        n_mc_samples_for_gradient_estimation : Number of monte carlo samples used for
            approximation of objective gradients
        method_as_string : str
            * "fullrank_advi":  we learn the correlations across all the betas.
            * "advi" : The default rather than fullrank_advi, because we want to compare to IB-CAVI, and our IB-CAVI
                model learns correlations across covariates for a given category, but assumes independence
                across categories.  In constrast, the FullRank approach of pymc3 would seem to learn
                correlations across all [K(M+1)]^2 entries -- which is a huge ask.  Given a choice
                between the two extremes of a FullRank approach and a complete mean field approach (where
                we assume independence across all the betas), we prefer the latter as a more conservative
                estimate of the run-time advantage of IB-CAVI.   When the simulated data has no correlations
                across betas, this should also give IB-CAVI a modeling advantage as well.
            * "mcmc"

    Returns:
        fitted advi model, intermediate_results

    Reference:
        https://docs.pymc.io/en/v3/pymc-examples/examples/generalized_linear_models/GLM.html
    """
    advi = model.fit(
        method=method_as_string, n_init=n_init, init=init, random_seed=random_seed
    )
    if my_intermediate_result_callback is not None:
        tracker = pymc3.callbacks.Tracker(
            intermediate_results_includes_Nones=my_intermediate_result_callback
        )
        model_fitted = advi.fit(n_its, callbacks=[tracker])
        intermediate_results = list(
            filter(None, tracker["intermediate_results_includes_Nones"])
        )  # filter out the Nones
        return model_fitted, intermediate_results
    else:
        model_fitted = advi.fit(n_its, obj_n_mc=n_mc_samples_for_gradient_estimation)
        return model_fitted, None


def force_bambi_model_to_have_standard_normal_priors(
    model: bmb.models.Model,
) -> bmb.models.Model:
    """
    We make the priors look like the ones used in our IB-CAVI script, i.e.
        beta_{mk} ~ N(0,1)    for covariates m and categories k
    so that the models are more comparable.

    By default (which we override), Bambi sets a reference prior that is probably smarter.
    TODO: Read up on it and see if we want to use that across the board instance (including in IB-CAVI).
    """
    # make the priors look like the one used by IB-CAVI, so the models are more comparable.
    model.set_priors(common=bmb.Prior("Normal", mu=0.0, sigma=1.0))
    model.set_priors(priors={"Intercept": bmb.Prior("Normal", mu=0.0, sigma=1.0)})
    return model


def get_beta_mean_from_variational_approximation(
    approximation: MeanField,
    labels_train: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        labels_train:  an np.array of shape (n_samples, n_categories);
            each row is one-hot encoded.  We need this because the ADVI
            training will only return beta vectors for the categories that were observed.
    Returns:
        the variational mean beta matrix, a matrix of shape (M,K),
            where M is the number of covariates, and K is the number of categories.
            Note that ADVI fits an --identified-- softmax, so the beta vector for the 0th category is
            predetermined to be the zero vector.
    """
    warnings.warn(
        f"We are assuming the category assigned to have a mean of 0 and variance of 0 "
        f"(for identification) is the category labeled 0.  "
        f"Hard to know at this point how to confirm this using pymc3/bambi. "
        f"Ideally this would be done programtically."
    )

    posterior_means_as_vector = approximation.params[0].eval()
    posterior_means_as_dict = approximation.bij.rmap(posterior_means_as_vector)
    return _get_beta_mean_from_posterior_means_as_dict(
        posterior_means_as_dict, labels_train
    )


def get_beta_mean_from_flat_beta_vector(
    flat_beta_vector: NumpyArray1D,
    fitted_advi_model: MeanField,
    labels_train: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        labels_train:  an np.array of shape (n_samples, n_categories);
            each row is one-hot encoded.  We need this because the ADVI
            training will only return beta vectors for the categories that were observed.
    Returns:
        the variational mean beta matrix, a matrix of shape (M,K),
            where M is the number of covariates, and K is the number of categories.
            Note that ADVI fits an --identified-- softmax, so the beta vector for the 0th category is
            predetermined to be the zero vector.
    """
    warnings.warn(
        f"We are assuming the category assigned to have a mean of 0 and variance of 0 "
        f"(for identification) is the category labeled 0.  "
        f"Hard to know at this point how to confirm this using pymc3/bambi. "
        f"Ideally this would be done programtically."
    )

    posterior_means_as_dict = fitted_advi_model.bij.rmap(flat_beta_vector)
    return _get_beta_mean_from_posterior_means_as_dict(
        posterior_means_as_dict, labels_train
    )


def _get_beta_mean_from_posterior_means_as_dict(
    posterior_means_as_dict: Dict[str, NumpyArray1D],
    labels_train: NumpyArray2D,
) -> NumpyArray2D:
    """
    posterior_means_as_dict:
        Example:
            {'cov0': array([0.313, 0.376]),
            'Intercept': array([-0.041, -0.34 ]),
            'cov2': array([-0.578,  0.5  ]),
            'cov1': array([0.509, 0.511])}
    """

    # for standard deviations, do:
    # sds_pymc3 = approximation.std.eval()
    # pymc3_sd_dict = approximation.bij.rmap(sds_pymc3)

    K, M_including_intercept = np.shape(labels_train)[1], len(
        posterior_means_as_dict.keys()
    )
    M = M_including_intercept - 1
    beta_mean_ADVI = np.zeros((M_including_intercept, K))

    # We need to make sure the beta matrix has the right dimension (M,K), and not
    # (M,J) for some J<K, which can happen if some categories weren't observed in the training set.
    # TODO: find the categories that weren't observed, and give those the value of the prior mean more explicitly.
    # This code implicitly assumes the prior mean was 0.
    choices_train = np.argmax(labels_train, 1)
    train_cats_observed = sorted(list(set(choices_train)))
    if 0 not in train_cats_observed:
        raise NotImplementedError("Need to update code to handle this case")
    # we don't care about zeroth category due to identifiability.
    # Bambi's softmax model seems to fix the beta vector for the 0th category at the zero vector.
    # But ideally we shouldn't make this assumption, and should proceed programmatically.
    train_cats_observed_without_zeroth_category = train_cats_observed[1:]

    intercept_mean = posterior_means_as_dict["Intercept"]
    beta_mean_ADVI[0, train_cats_observed_without_zeroth_category] = intercept_mean
    for m in range(M):
        beta_mean_ADVI[
            m + 1, train_cats_observed_without_zeroth_category
        ] = posterior_means_as_dict[f"cov{m}"]
    return beta_mean_ADVI
