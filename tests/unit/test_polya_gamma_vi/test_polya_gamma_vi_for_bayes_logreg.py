import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_logistic_regression_dataset,
)
from categorical_from_binary.polya_gamma.binary_logreg_vi.inference import (
    PriorParameters,
    compute_kl_divergence_from_prior_to_variational_beta,
    run_polya_gamma_variational_inference_for_bayesian_logistic_regression,
)


def test_run_polya_gamma_variational_inference_for_bayesian_logistic_regression():
    """
    Check our polya gamma VI algorithm for bayesian logistic regrression.
    In particular, we check that the variational mean for the regression weights is similar to the regression weights
    provided by sklearn.
    """
    dataset = generate_logistic_regression_dataset(n_samples=1000, n_features=5)

    # set up prior
    beta_dim = len(dataset.beta)
    prior_mean_beta = np.zeros(beta_dim)
    prior_cov_beta = np.eye(beta_dim)
    prior_params = PriorParameters(prior_mean_beta, prior_cov_beta)

    variational_params = (
        run_polya_gamma_variational_inference_for_bayesian_logistic_regression(
            dataset,
            prior_params,
            verbose=True,
            convergence_criterion_drop_in_elbo=0.01,
        )
    )
    variational_mean_beta = variational_params.mean_beta

    lr = LogisticRegression(random_state=0).fit(dataset.features, dataset.labels)
    sklearn_beta = np.concatenate((lr.intercept_, lr.coef_[0, :]))

    assert np.isclose(variational_mean_beta, sklearn_beta, atol=0.01, rtol=0.05).all()


def test_that_compute_kl_divergence_from_prior_to_variational_beta_is_zero_when_distributions_are_identical():

    dim = 10  # arbitary choice

    random_mean = scipy.stats.multivariate_normal(mean=np.zeros(dim)).rvs()
    random_cov = scipy.stats.invwishart(df=dim, scale=np.eye(dim)).rvs()

    mean_list = [np.zeros(dim), random_mean]
    cov_list = [np.eye(dim), random_cov]

    for mean, cov in zip(mean_list, cov_list):
        expected_kl = 0
        computed_kl = compute_kl_divergence_from_prior_to_variational_beta(
            mean,
            cov,
            mean,
            cov,
        )
        assert np.isclose(computed_kl, expected_kl, atol=1e-5)


# TODO: Write test that ELBO decreases over inference.  We see it in the printouts, but could test this automatically.
# To do this, we'd need the main inference function (or some helper function) to return the ELBOs, or at least to
# return the history of variational parameters
