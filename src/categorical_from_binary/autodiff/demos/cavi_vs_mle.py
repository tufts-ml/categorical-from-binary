"""
Why?

1. Can get MLE of CBM and MLE of IB to see if they line up as n->infty 
2. Want to get variance in a category prob as a fucntion of the variance in beta.


For now let's just use binary CBM.
Reference:  https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf

"""
import jax.numpy as np


np.set_printoptions(precision=3, suppress=True)


from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions as compute_CBC_Probit_predictions_with_autodiff_MLE,
    compute_CBM_Probit_predictions as compute_CBM_Probit_predictions_with_autodiff_MLE,
    compute_IB_Probit_predictions as compute_IB_Probit_predictions_with_autodiff_MLE,
    optimize_beta_for_CBC_model,
    optimize_beta_for_CBM_model,
    optimize_beta_for_IB_model,
)
from categorical_from_binary.autodiff.simplex_distances import compute_mean_l1_distance
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_cbc_probit_probabilities,
    construct_cbm_probit_probabilities,
    construct_multi_logit_probabilities,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


###
# Construct dataset
###
n_categories = 3
n_features = 1
n_samples = 5000
include_intercept = True
link = Link.MULTI_LOGIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link,
    seed=None,
    include_intercept=include_intercept,
)

# try MLE
beta_init = np.zeros((n_features + 1, n_categories))

beta_star_CBC_MLE = optimize_beta_for_CBC_model(
    beta_init, dataset.features, dataset.labels
)
category_probs_CBC_MLE = compute_CBC_Probit_predictions_with_autodiff_MLE(
    beta_star_CBC_MLE, dataset.features
)


beta_star_CBM_MLE = optimize_beta_for_CBM_model(
    beta_init, dataset.features, dataset.labels
)
category_probs_CBM_MLE = compute_CBM_Probit_predictions_with_autodiff_MLE(
    beta_star_CBM_MLE, dataset.features
)

beta_star_IB_MLE = optimize_beta_for_IB_model(
    beta_init, dataset.features, dataset.labels
)
category_probs_IB_MLE = compute_IB_Probit_predictions_with_autodiff_MLE(
    beta_star_IB_MLE, dataset.features
)


category_probs_CBC_with_IB_MLE = construct_cbc_probit_probabilities(
    dataset.features, beta_star_IB_MLE
)

category_probs_CBC_with_CBC_MLE = construct_cbc_probit_probabilities(
    dataset.features, beta_star_CBC_MLE
)

category_probs_CBM_with_IB_MLE = construct_cbm_probit_probabilities(
    dataset.features, beta_star_IB_MLE
)

category_probs_CBM_with_CBM_MLE = construct_cbm_probit_probabilities(
    dataset.features, beta_star_CBM_MLE
)

category_probs_true = construct_multi_logit_probabilities(
    dataset.features, dataset.beta
)


# Comparing IB beta to CBM/CBC beta is hard becuase of non-identifiability.  Very different betas
# can give essentially the same category probs.  An example:  beta_star_IB_MLE and beta_star_CBM_MLE
# are very different, but they give essentially the same CBM category probs.
#
# To get around this problem while still comparing CBM to CBC, we could perhaps propose that
# Difference_in_some_sense( CBM_probs(beta_star_IB_MLE), CBM_probs(beta_star_CBM_MLE) )  is less than
# Difference_in_some_sense( CBC_probs(beta_star_IB_MLE), CBC_probs(beta_star_CBC_MLE) )


#### Comparison 1
error_to_MLE_probs_CBC = compute_mean_l1_distance(
    category_probs_CBC_with_IB_MLE, category_probs_CBC_with_CBC_MLE
)
error_to_MLE_probs_CBM = compute_mean_l1_distance(
    category_probs_CBM_with_IB_MLE, category_probs_CBM_with_CBM_MLE
)
print(
    f"Error using MLE from IB instead of actual MLE. CBM: {error_to_MLE_probs_CBM :.03f}, CBC: {error_to_MLE_probs_CBC:.03f}"
)
# results sometmes favor CBC, sometimes CBM.  note: when CBC is off, it tends to sharpen the
# category probabilities.  when CBM is off, it tends to soften the category probabilities.


### Comparison 2
error_to_ground_truth_probs_CBC = compute_mean_l1_distance(
    category_probs_CBC_with_IB_MLE, category_probs_true
)
error_to_ground_truth_probs_CBM = compute_mean_l1_distance(
    category_probs_CBM_with_IB_MLE, category_probs_true
)
print(
    f"Error compared to ground truth. CBM: {error_to_ground_truth_probs_CBM :.03f}, CBC: {error_to_ground_truth_probs_CBC :.03f}"
)
# TODO: show category probs for CAVI -> category probs for MLE as n to infinity.  just for funsies / sanity check.


###
# CAVI
###
results = compute_multiclass_probit_vi_with_normal_prior(
    dataset.labels,
    dataset.features,
    variational_params_init=None,
    convergence_criterion_drop_in_mean_elbo=0.001,
)
beta_CAVI_mean = results.variational_params.beta.mean


category_probs_CBC_with_IB_CAVI_mean = construct_cbc_probit_probabilities(
    dataset.features, beta_CAVI_mean
)

category_probs_CBM_with_IB_CAVI_mean = construct_cbm_probit_probabilities(
    dataset.features, beta_CAVI_mean
)

### Comparison 3
error_CBC = np.sum(
    np.abs(category_probs_CBC_with_IB_CAVI_mean - category_probs_CBC_with_IB_MLE)
)
error_CBM = np.sum(
    np.abs(category_probs_CBM_with_IB_CAVI_mean - category_probs_CBM_with_IB_MLE)
)
# note: this comparison is dumb.  IB_CAVI will converge to IB_MLE, we're just plugging these into different formulae...

### Comparison 4
error_CBC = np.sum(
    np.abs(category_probs_CBC_with_IB_CAVI_mean - category_probs_CBC_with_CBC_MLE)
)
error_CBM = np.sum(
    np.abs(category_probs_CBM_with_IB_CAVI_mean - category_probs_CBM_with_CBM_MLE)
)
# this is just like comparison 1.  I think again it's because IB_CAVI converges to IB_MLE.


"""
Questions:
    * Compare beta for {CBM MLE, CBM CAVI} to IB MLE.   Does LHS converge to RHS?   Does this NOT happen for CBC?  
    * Exploratory: Do rates of convergence differ for CBM and CBC?
"""
