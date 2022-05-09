"""
References:
    Wojnowicz, Michael.  Categorical models with variational inference.  Available upon request.
"""

from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    HierarchicalMulticlassRegressionDataset,
)
from categorical_from_binary.ib_cavi.binary_probit.elbo import (
    compute_variational_entropy_of_z as compute_variational_entropy_of_z_for_one_group_and_category,
    compute_variational_expectation_of_complete_data_likelihood as compute_variational_expectation_of_complete_data_likelihood_for_one_group_and_category,
)
from categorical_from_binary.ib_cavi.multi.hierarchical_ib_probit.structs import (
    Hyperparameters,
    VariationalParams,
    data_dimensions_from_hierarchical_dataset,
)
from categorical_from_binary.kl import (
    compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument,
    compute_kl_inverse_wishart,
    compute_kl_mvn,
)


# TODO: A lot of these functions involve running through loops.  They will therefore
# be slow in Python, although not in C or Julia.  Need to either vectorize (annoying)
# or compile to C with Cython, or port to Julia, or something.


def compute_variational_expectation_of_complete_data_likelihood(
    data: HierarchicalMulticlassRegressionDataset,
    vp: VariationalParams,
):
    dims = data_dimensions_from_hierarchical_dataset(data)
    ecdl = 0  # expected complete data likelihood
    for j in range(dims.num_groups):
        covariates = data.datasets[j].features
        for k in range(dims.num_categories):
            z_mean = vp.zs_expected[j][:, k]
            beta_mean = vp.beta_means[j, :, k]
            beta_cov = vp.beta_covs[j, :, :, k]
            ecdl += compute_variational_expectation_of_complete_data_likelihood_for_one_group_and_category(
                z_mean, beta_mean, beta_cov, covariates
            )
    return ecdl


def compute_variational_entropy_of_z(
    data: HierarchicalMulticlassRegressionDataset,
    vp: VariationalParams,
) -> float:
    dims = data_dimensions_from_hierarchical_dataset(data)
    vez = 0  # variational entropy of z
    for j in range(dims.num_groups):
        covariates = data.datasets[j].features
        for k in range(dims.num_categories):
            beta_mean = vp.beta_means[j, :, k]
            labels = data.datasets[j].labels[:, k]
            vez += compute_variational_entropy_of_z_for_one_group_and_category(
                labels, covariates, beta_mean
            )
    return vez


def compute_kl_divergence_of_mus(
    data: HierarchicalMulticlassRegressionDataset,
    hp: Hyperparameters,
    vp: VariationalParams,
) -> float:
    """
    Remarks:
        The data is not actually needed for the computation; we just use it to extract the data dimensions
    """
    dims = data_dimensions_from_hierarchical_dataset(data)
    kl_mus = 0
    for k in range(dims.num_categories):
        mu_mean_variational = vp.mu_means[:, k]
        mu_cov_variational = vp.mu_covs[:, :, k]
        mu_mean_prior = hp.m_0
        mu_cov_prior = hp.V_0
        kl_mus += compute_kl_mvn(
            mu_mean_variational, mu_cov_variational, mu_mean_prior, mu_cov_prior
        )
    return kl_mus


def compute_kl_divergence_of_Sigmas(
    data: HierarchicalMulticlassRegressionDataset,
    hp: Hyperparameters,
    vp: VariationalParams,
) -> float:
    """
    Remarks:
        The data is not actually needed for the computation; we just use it to extract the data dimensions
    """
    dims = data_dimensions_from_hierarchical_dataset(data)
    kl_Sigmas = 0
    for k in range(dims.num_categories):
        Sigma_df_variational = vp.Sigma_dfs[k]
        Sigma_RSS_variational = vp.Sigma_RSSs[:, :, k]
        Sigma_df_prior = hp.nu_0
        Sigma_RSS_prior = hp.S_0
        kl_Sigmas += compute_kl_inverse_wishart(
            Sigma_df_variational, Sigma_RSS_variational, Sigma_df_prior, Sigma_RSS_prior
        )
    return kl_Sigmas


def compute_variational_expectation_of_kl_divergence_of_betas(
    data: HierarchicalMulticlassRegressionDataset,
    vp: VariationalParams,
) -> float:
    """
    Remarks:
        The data is not actually needed for the computation; we just use it to extract the data dimensions
    References:
        Wojnowicz, Michael.  Categorical models with variational inference.  Available upon request.
    """
    dims = data_dimensions_from_hierarchical_dataset(data)

    vekl = 0  # variational expectation of kl divergence

    for k in range(dims.num_categories):
        for j in range(dims.num_groups):
            beta_mean_variational = vp.beta_means[j, :, k]
            beta_cov_variational = vp.beta_covs[j, :, :, k]
            mu_mean_variational = vp.mu_means[:, k]
            mu_cov_variational = vp.mu_covs[:, :, k]
            Sigma_df_variational = vp.Sigma_dfs[k]
            Sigma_RSS_variational = vp.Sigma_RSSs[:, :, k]
            vekl += compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument(
                beta_mean_variational,
                beta_cov_variational,
                mu_mean_variational,
                mu_cov_variational,
                Sigma_df_variational,
                Sigma_RSS_variational,
            )
    return vekl


def compute_elbo(
    data: HierarchicalMulticlassRegressionDataset,
    hp: Hyperparameters,
    vp: VariationalParams,
) -> float:

    return (
        compute_variational_expectation_of_complete_data_likelihood(data, vp)
        + compute_variational_entropy_of_z(data, vp)
        - compute_kl_divergence_of_mus(data, hp, vp)
        - compute_kl_divergence_of_Sigmas(data, hp, vp)
        - compute_variational_expectation_of_kl_divergence_of_betas(data, vp)
    )
