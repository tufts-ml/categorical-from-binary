from sklearn.datasets import load_iris

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
# Data
###

data = load_iris()

# binary dataset
#   we make it a binary dataset by only using the first 2/3 classes
features = data.data[:100]
labels = data.target[:100]

# add intercept
covariates = prepend_features_with_column_of_all_ones_for_intercept(features)

####
# Inference
####

# info
variational_params = compute_probit_vi_with_normal_prior(
    labels,
    covariates,
    variational_params_init=None,
    max_n_iterations=20,
)
variational_beta = variational_params.beta
print(f"The variational mean for the regression weights is {variational_beta.mean}")

# sanity check...
class_probs = compute_class_probabilities(covariates, variational_beta.mean)
print_table_comparing_class_probs_to_labels(class_probs, labels)
