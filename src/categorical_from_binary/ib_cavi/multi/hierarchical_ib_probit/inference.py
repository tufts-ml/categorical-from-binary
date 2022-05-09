"""
The hierarchical multiclass regression model extends the CBC-Probit 
to be a multi-level (i.e. hierarchical) model, appropriate for grouped data.

Here we do coordinate ascent variational inference (CAVI)
"""
from pprint import pprint
from typing import List, Optional, Tuple

import numpy as np

from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    HierarchicalMulticlassRegressionDataset,
)
from categorical_from_binary.ib_cavi.multi.hierarchical_ib_probit.elbo import compute_elbo
from categorical_from_binary.ib_cavi.multi.hierarchical_ib_probit.structs import (
    Hyperparameters,
    VariationalParams,
    data_dimensions_from_hierarchical_dataset,
)
from categorical_from_binary.ib_cavi.trunc_norm import (
    compute_expected_value_normal_minus,
    compute_expected_value_normal_plus,
)
from categorical_from_binary.types import (
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    NumpyArray4D,
)


def construct_default_hyperparameters(num_designed_features: int):
    M = num_designed_features
    return Hyperparameters(V_0=np.eye(M), m_0=np.zeros(M), nu_0=M, S_0=M * np.eye(M))


def initialize_variational_params(
    data: HierarchicalMulticlassRegressionDataset,
    hp: Hyperparameters,
) -> VariationalParams:

    # TODO: could perhaps get a better initialization in a number of ways
    #   - Use some sort of ML/MAP scheme
    #   - Use a "cascading" init, where instead of using the prior for everything,
    #   we use accumulating info as we go along.  This would basically just be like a partial
    #   iteration of VI, though, so probably not much gain here, unless a single iteration of VI
    #   were very time consuming, and the initialization as well (very doubtful)

    # TODO: We're probably not really using -all- these initializations in practice.
    # E.g. if we initialize the beta_means to 0's, the first round of VI will give us non-zero
    # values for the expected_zs.... It might be good to separate out the sufficient initialization
    # from the redundant initialization.  Note that it's not sufficient, though, to just seed the beta_mean to
    # somthing (like 0).  That's because, when doing the variational update to the beta_means,  the expected_zs
    # is no longer enough info to perform the update;  due to the hierarchical structure, we also need info about
    # the higher-level mu_k's and Sigma_k's to proceed, and that information can come from the prior.  So perhaps
    # the only redundancy here is initializing the z's.  We could set those to None, to highlight that the initialization
    # is not being used anywhere, but then this function would only work if the run_[...]_vi function cycled
    # through the variational factors in a particular order (namely, z's before beta's), and this function shouldn't
    # need to know what that function does.... so.... TL;DR -- the initializations are partially redundant,
    # but I'm just going to leave it for now.
    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # Initialize zs_expected
    #   We initialize to the 0 vector.
    zs_expected = [None] * J
    for j in range(J):
        zs_expected[j] = np.zeros((Ns[j], K))

    # initialize variational params on Sigma
    #   We initialize using the prior
    Sigma_dfs = np.ones(K) * hp.nu_0
    Sigma_RSSs = np.zeros((M, M, K))
    for k in range(K):
        Sigma_RSSs[:, :, k] = hp.S_0

    # initialize variational params on mu
    #   We initialize using the prior
    mu_means = np.zeros((M, K))
    for k in range(K):
        mu_means[:, k] = hp.m_0

    mu_covs = np.zeros((M, M, K))
    for k in range(K):
        mu_covs[:, :, k] = hp.V_0

    # initialize variational params on the betas
    expected_Sigma_k_precision = np.linalg.inv(hp.S_0) * hp.nu_0
    beta_covs = np.zeros((J, M, M, K))
    for j in range(J):
        X_j = data.datasets[j].features
        for k in range(K):
            beta_covs[j, :, :, k] = np.linalg.inv(
                expected_Sigma_k_precision + X_j.T @ X_j
            )

    # just initialize to all zeros, which is what we'd seem to get anyways if we initialize the z's
    # to zeros and if the prior expected mu_k's are all 0 (i.e. m_0=0, which it is, by default,
    # and in many similar models).
    beta_means = np.zeros((J, M, K))

    return VariationalParams(
        beta_means=beta_means,
        beta_covs=beta_covs,
        mu_means=mu_means,
        mu_covs=mu_covs,
        Sigma_dfs=Sigma_dfs,
        Sigma_RSSs=Sigma_RSSs,
        zs_expected=zs_expected,
    )


def optimize_zs(
    vp: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
) -> List[NumpyArray2D]:
    """
    Update the variational parameters on the factors q(z_ijk)
    Here, the variational parameters are given as E[z_{ijk}], rather than natural parameters.
    Also, they are represented as List[NumpyArray2D] instead of NumpyArray3D since the number of
    units i=1,...,N_j can differ across the groups j=1,...,J
    """
    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # form nus (a list with J elements; each element has shape (N_j,K))
    nus = [None] * J
    for j in range(J):
        X_j = data.datasets[j].features
        for i in range(Ns[j]):
            nus[j] = X_j @ vp.beta_means[j, :, :]

    # form zs_expected
    zs_expected = [np.zeros((n_j, K)) for n_j in Ns]
    for j in range(J):
        # TODO: compute the choices once, up front, and pass it around.
        labels_j = data.datasets[j].labels
        choices_j = np.argmax(labels_j, 1)  # index of selected category
        for i in range(Ns[j]):
            for k in range(K):
                if choices_j[i] == k:
                    zs_expected[j][i, k] = compute_expected_value_normal_plus(
                        nus[j][i, k]
                    )
                else:
                    zs_expected[j][i, k] = compute_expected_value_normal_minus(
                        nus[j][i, k]
                    )
    return zs_expected


def optimize_betas(
    vp: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
) -> Tuple[NumpyArray3D, NumpyArray4D]:
    """
    Update the variational parameters on the factors q(beta_jk)
    """
    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # update beta covariance and beta means
    beta_means = np.zeros((J, M, K))
    beta_covs = np.zeros((J, M, M, K))
    for j in range(J):
        X_j = data.datasets[j].features
        for k in range(K):
            expected_Sigma_k_precision = (
                np.linalg.inv(vp.Sigma_RSSs[:, :, k]) * vp.Sigma_dfs[k]
            )
            beta_cov_jk = np.linalg.inv(expected_Sigma_k_precision + X_j.T @ X_j)
            beta_mean_jk = beta_cov_jk @ (
                expected_Sigma_k_precision @ vp.mu_means[:, k]
                + X_j.T @ vp.zs_expected[j][:, k]
            )
            beta_means[j, :, k] = beta_mean_jk
            beta_covs[j, :, :, k] = beta_cov_jk
    return beta_means, beta_covs


def optimize_mus(
    vp: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
    hp: Hyperparameters,
) -> Tuple[NumpyArray2D, NumpyArray3D]:
    """
    Update the variational parameters on the factors q(mu_k)
    """
    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # update mu covariance and mean
    mu_means = np.zeros((M, K))
    mu_covs = np.zeros((M, M, K))
    for k in range(K):
        expected_Sigma_k_precision = (
            np.linalg.inv(vp.Sigma_RSSs[:, :, k]) * vp.Sigma_dfs[k]
        )
        beta_mean_k_summed_across_j = np.sum(vp.beta_means, 0)[:, k]
        # TODO: take the inverse of the hyperparameter only once, at the beginning
        mu_cov_k = np.linalg.inv(np.linalg.inv(hp.V_0) + J * expected_Sigma_k_precision)
        mu_mean_k = mu_cov_k @ (
            np.linalg.inv(hp.V_0) @ hp.m_0
            + expected_Sigma_k_precision @ beta_mean_k_summed_across_j
        )
        mu_means[:, k] = mu_mean_k
        mu_covs[:, :, k] = mu_cov_k
    return mu_means, mu_covs


def optimize_Sigmas(
    vp: VariationalParams,
    data: HierarchicalMulticlassRegressionDataset,
    hp: Hyperparameters,
) -> Tuple[NumpyArray1D, NumpyArray3D]:
    """
    Update the variational parameters on the factors q(Sigma_k)
    """
    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # update Sigma dfs and Residual Sums of Squares
    Sigma_dfs = np.zeros(K)
    Sigma_RSSs = np.zeros((M, M, K))
    for k in range(K):
        Sigma_df_k = hp.nu_0 + J
        Sigma_RSS_k = (
            hp.S_0
            + J * vp.mu_covs[:, :, k]
            + np.sum(vp.beta_covs, 0)[:, :, k]
            + sum(
                np.outer(
                    vp.beta_means[j, :, k] - vp.mu_means[:, k],
                    vp.beta_means[j, :, k] - vp.mu_means[:, k],
                )
                for j in range(J)
            )
        )
        Sigma_dfs[k] = Sigma_df_k
        Sigma_RSSs[:, :, k] = Sigma_RSS_k
    return Sigma_dfs, Sigma_RSSs


def run_hierarchical_multiclass_probit_vi(
    data: HierarchicalMulticlassRegressionDataset,
    hyperparams: Optional[Hyperparameters] = None,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_elbo: float = -np.inf,
    variational_params_init=None,
    use_autoregressive_design_matrix: bool = False,
) -> VariationalParams:
    """
    Variational inference for Bayesian hierarchical multiclass probit regression.
    """
    if variational_params_init is not None:
        raise NotImplementedError(
            "Haven't yet updated this from the non-hierarchical case."
        )

    if use_autoregressive_design_matrix:
        raise NotImplementedError

    # get data dimensionality
    J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)

    # construct hyperparameters
    if hyperparams is None:
        hyperparams = construct_default_hyperparameters(M)
    print("Hyperparameters:")
    pprint(hyperparams._asdict())

    # initialize variational params
    vp = initialize_variational_params(data, hyperparams)

    # coordinate-ascent variational inference
    n_iterations_so_far = 0
    previous_elbo, drop_in_elbo = -np.inf, np.inf  # noqa
    print(
        f"Max # iterations: {max_n_iterations}.  Convergence criterion (drop in ELBO): {convergence_criterion_drop_in_elbo}"
    )
    while (
        n_iterations_so_far <= max_n_iterations
        and drop_in_elbo >= convergence_criterion_drop_in_elbo
    ):
        vp.zs_expected = optimize_zs(vp, data)
        vp.beta_means, vp.beta_covs = optimize_betas(vp, data)
        vp.mu_means, vp.mu_covs = optimize_mus(vp, data, hyperparams)
        vp.Sigma_dfs, vp.Sigma_RSSs = optimize_Sigmas(vp, data, hyperparams)

        # print(f"Expected population-level beta: \n {vp.mu_means}")

        # ELBO computation..
        elbo = compute_elbo(data, hyperparams, vp)
        drop_in_elbo = elbo - previous_elbo
        previous_elbo = elbo
        n_iterations_so_far += 1

        # TODO:  Adjust this code to the hierarchical setting; print out mean selection prob
        # # running category probs
        # category_probs = construct_cbc_probit_probabilities(
        #     design_matrix, beta_mean
        # )
        # category_probs_of_choices = [
        #     category_probs[i, choice] for (i, choice) in enumerate(choices)
        # ]
        # mean_selection_prob = np.mean(category_probs_of_choices)

        print(f"Iteration: {n_iterations_so_far}  ELBO: {elbo:.06}")

    return vp
