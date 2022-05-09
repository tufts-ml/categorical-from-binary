import numpy as np

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_logistic_regression_dataset,
)
from categorical_from_binary.laplace_vi.bayes_logreg.inference import (
    optimize_beta_for_bayesian_logreg_using_laplace_vi,
)


def test_optimize_beta_for_bayesian_logreg_using_laplace_vi():
    print(
        f"\n\nNow testing Laplace Variational Inference with Bayesian Logistic Regression \n"
        f"In particular, is the posterior mean beta close to the true generating regression weights for n large?"
    )
    n_samples, n_features = 10000, 5
    dataset = generate_logistic_regression_dataset(n_samples, n_features, seed=1)
    beta_init = np.zeros(n_features + 1)
    prior_mean = np.zeros(n_features + 1)
    prior_cov = np.eye(n_features + 1)

    beta_fitted = optimize_beta_for_bayesian_logreg_using_laplace_vi(
        beta_init,
        dataset,
        prior_mean,
        prior_cov,
    )
    beta_true = dataset.beta
    print(
        f"Num samples: {n_samples} \n Beta fitted: {beta_fitted} \n Beta true: {beta_true}"
    )
    assert np.allclose(beta_fitted, beta_true, rtol=0.25)
