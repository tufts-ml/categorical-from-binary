"""
In this script, we explore the properties of our code in inference.py which does
variational Inference for Bayesian multiclass logistic regression with 
* stick-breaking link function
* polya-gamma augmentation (for conditional conjugacy)

In particular, as of 3/15/2021, our demo reveals that the code is not functioning
as well as we would have hoped, so we try to understand why.
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


# """
# Does permutation (ordered by most common labels in the dataset) help?
# """
# label_idxs_ordered_by_popularity=np.flip(np.argsort(sum(dataset.labels,1))) #most to least popular

# MulticlassRegressionDataset(dataset.features, dataset.labels[:,label_idxs_ordered_by_popularity], dataset.beta[:,label_idxs_ordered_by_popularity[:-1]])


"""
Q: Does the quality of inference depend on the ORDER of the frequencies of the categories?

We will explore this question by hardcoding beta_0 in our data generation scheme. 
"""


ORDER_CATEGORIES_BY_ASCENDING_POPULARITY = True  # If True, will get better results
n_categories = 4

# The last (4th) category always has its beta vector (including beta_0) set to 0.
beta_0_for_ordering_categories_from_least_to_most_popular = [-3, -2, -1]
beta_0_for_ordering_categories_from_most_to_least_popular = [3, 2, 1]

if ORDER_CATEGORIES_BY_ASCENDING_POPULARITY:
    beta_0 = beta_0_for_ordering_categories_from_least_to_most_popular
else:
    beta_0 = beta_0_for_ordering_categories_from_most_to_least_popular

dataset = generate_multiclass_regression_dataset(
    n_samples=10000, n_features=5, n_categories=n_categories, beta_0=beta_0, seed=30
)
print(f"The number of times each category label was used: {np.sum(dataset.labels,0)}")

beta_dim_per_category, n_free_categories = np.shape(dataset.beta)
prior_means_beta = [np.zeros(beta_dim_per_category)] * n_free_categories
prior_covs_beta = [np.eye(beta_dim_per_category)] * n_free_categories
prior_params = PriorParameters_MulticlassLogisticRegression(
    prior_means_beta, prior_covs_beta
)

variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
        dataset,
        prior_params,
        variational_params_init=None,
        max_n_iterations=20,
        verbose=True,
    )
)

### compare betas
print(f"Comparing true beta to variational mean")
print(f"true: \n {dataset.beta}")
print(f"variational: \n {np.transpose(variational_params.means_beta)}")

### compare category probs
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
print("Category probs (training set)")
print(f"true: \n {true_category_probs}")
print(f"sklearn: \n {sklearn_category_probs}")
print(f"variational: \n {variational_category_probs}")


# crappy error comparisons, this isn't respecting the fact that the rows are on the simplex
sum_absolute_error_sklearn = np.sum(abs(true_category_probs - sklearn_category_probs))
sum_absolute_error_variational = np.sum(
    abs(true_category_probs - variational_category_probs)
)
print("Sum of absolute error in Predictive probs")
print(f"sklearn: \n {sum_absolute_error_sklearn}")
print(f"variational: \n {sum_absolute_error_variational}")


"""
A: Yes!  Very much so!  We need to order the categories by ascending popularity (least to most)


ORDER_CATEGORIES_BY_ASCENDING_POPULARITY = False
Comparing true beta to variational mean
true: 
 [[ 3.     2.     1.   ]
 [ 1.332  0.715 -1.545]
 [-0.008  0.621 -0.72 ]
 [ 0.266  0.109  0.004]
 [-0.175  0.433  1.203]
 [-0.965  1.028  0.229]]
variational: 
 [[ 0.07   0.527  1.058]
 [ 1.348  1.91  -1.531]
 [-0.141  1.188 -0.707]
 [ 0.173  0.074 -0.006]
 [-0.795 -0.508  1.384]
 [-1.555  0.927  0.2  ]]

Sum of absolute error in Predictive probs
sklearn: 
 202.50153036300685
variational: 
 5542.556943937899

ORDER_CATEGORIES_BY_ASCENDING_POPULARITY = True

Comparing true beta to variational mean
true: 
 [[-3.    -2.    -1.   ]
 [ 1.332  0.715 -1.545]
 [-0.008  0.621 -0.72 ]
 [ 0.266  0.109  0.004]
 [-0.175  0.433  1.203]
 [-0.965  1.028  0.229]]
variational: 
 [[-3.535 -2.51  -0.944]
 [ 1.34   1.007 -1.518]
 [-0.035  0.766 -0.704]
 [ 0.153  0.005 -0.019]
 [-0.372  0.181  1.157]
 [-1.108  1.003  0.247]]


Sum of absolute error in Predictive probs
sklearn: 
 247.50099382500508
variational: 
 1116.6039000131668
"""

"""
Rk: We are still not matching scikit-learn, though, even when we have a favorable ordering.
* Would we eventually match it if we pushed the order assymmetry further?  Maybe so -- error reduced
to double of sckit learn when the beta_0's were [-6, -4, -2, 0]
* Is this because of our assumption that betas are independent across categories?
* Is there something else we're doing that is suboptimal?
"""

"""
Rk: I would have expected the code to work best when the categories were ordered from MOST popular to least popular,
not the other way around.  
"""
