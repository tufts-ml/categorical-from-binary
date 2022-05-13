"""
In this script, we demo / sanity check the code in inference.py, which does
variational Inference for Bayesian multiclass logistic regression with 
* stick-breaking link function
* polya-gamma augmentation (for conditional conjugacy)

There are currently problems with inference, as shown here.  For more info, see
exploration_of_label_asymmetry.py
"""

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_multi_logit_probabilities,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.evaluate.multiclass import (
    get_sklearn_category_probabilities,
)
from categorical_from_binary.polya_gamma.stickbreak_multiclass_logreg_vi.inference import (
    PriorParameters_MulticlassLogisticRegression,
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression,
)


"""
Q: Does the variational mean in our scheme approximate the truth?
"""

# generate data
dataset = generate_multiclass_regression_dataset(
    n_samples=10000, n_features=5, n_categories=4, seed=10
)


# construct prior
beta_dim_per_category, n_free_categories = np.shape(dataset.beta)
prior_means_beta = [np.zeros(beta_dim_per_category)] * n_free_categories
prior_covs_beta = [np.eye(beta_dim_per_category)] * n_free_categories
prior_params = PriorParameters_MulticlassLogisticRegression(
    prior_means_beta, prior_covs_beta
)

# run variational inference
variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
        dataset,
        prior_params,
        variational_params_init=None,
        max_n_iterations=20,
        verbose=True,
    )
)

###
# Inspect results
###

### compare variational mean betas to true generative betas
print(f"\n\n Comparing true beta to variational mean")
print(f"true: \n {dataset.beta}")
print(f"variational: \n {np.transpose(variational_params.means_beta)}")

"""
A:  Yes and No.  Betas for later categories are closer to the truth. 

Comparing true beta to variational mean when seed=5 for 
> dataset = generate_multiclass_regression_dataset(n_samples=10000, n_features=5, n_categories=3, seed=5)


true: [[ 0.11  -0.083]
 [ 2.431 -0.252]
 [ 0.11   1.582]
 [-0.909 -0.592]
 [ 0.188 -0.33 ]
 [-1.193 -0.205]]
variational: [[-0.806 -0.024]
 [ 2.485 -0.235]
 [-0.646  1.616]
 [-0.622 -0.625]
 [ 0.342 -0.345]
 [-1.04  -0.219]]

If I vary the number of categories, the phenomenon seems to always hold, where the last one is better estimated than the others.
"""


### Do the training set category probabilities match up with sklearn?
true_category_probs = construct_multi_logit_probabilities(
    dataset.features, dataset.beta
)
variational_category_probs = construct_multi_logit_probabilities(
    dataset.features, np.transpose(variational_params.means_beta)
)
sklearn_category_probs = get_sklearn_category_probabilities(
    dataset.features, dataset.labels
)

np.set_printoptions(precision=3, suppress=True)
print("\n\nTraining set category probs")
print(f"true: \n {true_category_probs}")
print(f"sklearn: \n {sklearn_category_probs}")
print(f"variational: \n {variational_category_probs}")


### Compare the errors in the training set category probabilities with the errors using sklearn.
# crappy error comparisons, this isn't respecting the fact that the rows are on the simplex
sum_absolute_error_sklearn = np.sum(abs(true_category_probs - sklearn_category_probs))
sum_absolute_error_variational = np.sum(
    abs(true_category_probs - variational_category_probs)
)
print("\n\nSum of absolute error in Predictive probs")
print(f"sklearn: \n {sum_absolute_error_sklearn}")
print(f"variational: \n {sum_absolute_error_variational}")

"""
Sklearn is much closer than my variational code.

Sum of absolute error in Predictive probs
sklearn: 
 294.1383539850775
variational: 
 3717.5800003349027
"""
