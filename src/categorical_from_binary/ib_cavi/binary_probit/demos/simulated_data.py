import numpy as np

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_probit_regression_dataset,
)
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.ib_cavi.binary_probit.inference.main import (
    compute_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.binary_probit.sanity import (
    compute_class_probabilities,
    print_table_comparing_class_probs_to_labels,
)


###
# Demo -- generate data, do variational inference
###
dataset = generate_probit_regression_dataset(n_samples=1000, n_features=5, seed=10)
covariates = prepend_features_with_column_of_all_ones_for_intercept(dataset.features)
labels = dataset.labels

variational_params = compute_probit_vi_with_normal_prior(
    labels,
    covariates,
    variational_params_init=None,
    convergence_criterion_drop_in_elbo=0.1,
    max_n_iterations=np.inf,
)

variational_beta = variational_params.beta

###
# Sanity checks
###

# ### Sanity check -- show fitted category probabilities for some of the binary responses
print(
    f"Now printing fitted category probabilities for the selected response in a small sample"
)
n_samples = 10
covariates_sample, labels_sample = covariates[:n_samples], labels[:n_samples]
class_probs = compute_class_probabilities(covariates_sample, variational_beta.mean)
print_table_comparing_class_probs_to_labels(labels_sample, class_probs)

### Sanity check against "true" values of beta
print(
    f"The variational mean for the regression weights is \n {variational_beta.mean}.\n"
    f"The true value of the regression weights is \n {dataset.beta}."
)

# TODO: Add sanity check for the estimated variance of beta.
