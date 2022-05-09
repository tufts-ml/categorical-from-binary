import numpy as np
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_logistic_regression_dataset,
)
from categorical_from_binary.polya_gamma.binary_logreg_vi.inference import (
    PriorParameters,
    run_polya_gamma_variational_inference_for_bayesian_logistic_regression,
)


###
# Example
###
dataset = generate_logistic_regression_dataset(n_samples=10000, n_features=5)
beta_dim = len(dataset.beta)

prior_mean_beta = np.zeros(beta_dim)
prior_cov_beta = np.eye(beta_dim)
prior_params = PriorParameters(prior_mean_beta, prior_cov_beta)


variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_logistic_regression(
        dataset, prior_params, verbose=True, convergence_criterion_drop_in_elbo=0.01
    )
)

###
# Check betas against sklearn logistic regression
###

lr = LogisticRegression(random_state=0).fit(dataset.features, dataset.labels)
sklearn_betas = np.concatenate((lr.intercept_, lr.coef_[0, :]))

###
# Tabulate results
###

# tabulate reference: https://stackoverflow.com/questions/9712085/numpy-pretty-print-tabular-data

titles = ["variational_mean", "sklearn", "true"]
results = np.transpose([variational_params.mean_beta, sklearn_betas, dataset.beta])
table = tabulate(results, titles, tablefmt="fancy_grid")
print(table)
