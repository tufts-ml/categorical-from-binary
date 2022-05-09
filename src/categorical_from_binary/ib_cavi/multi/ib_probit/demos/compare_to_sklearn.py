"""
The model is the CBC-Probit model.
Here we do coordinate ascent variational inference (CAVI)

We show that the resulting model beats the multiclass regression model in
sklearn in terms of category probabilities of hold-out test set 
(possibly this `beating` is just due to it being lighter tailed..) 
"""

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.evaluate.multiclass import (
    Metric,
    evaluate_multiclass_regression_with_beta_estimate,
    evaluate_sklearn_on_multiclass_regression,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


###
# Construct dataset
###
n_categories = 3
n_features = 3
n_samples = 5000
include_intercept = True
link = Link.MULTI_LOGIT  # Link.CBC_PROBIT
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link,
    seed=None,
    beta_category_strategy=ControlCategoryPredictability(),
    include_intercept=include_intercept,
)


# Prep training data
n_train_samples = 4000
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
# `labels_train` gives one-hot encoded representation of category

####
# Variational Inference
####
results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    variational_params_init=None,
    convergence_criterion_drop_in_mean_elbo=0.01,
)
variational_params = results.variational_params
beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov

####
# Model Checking
####

# TODO: Re-add ability to compare to sklearn.
# WARNING: I don't think the covariates the same as the design matrix we used during inference
# in the case where use_autoregressive_design_matrix=True  for `compute_multiclass_probit_vi_with_normal_prior`.
# Possible solution:  pull `construct_design_matrix` out of inference and into preprocessing

print("Model check on training set...")
evaluate_multiclass_regression_with_beta_estimate(
    covariates_train,
    labels_train,
    beta_mean,
    link_for_category_probabilities=Link.CBM_PROBIT,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)
evaluate_sklearn_on_multiclass_regression(
    covariates_train,
    labels_train,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)

print("\nModel check on test set...")
# TODO: need to compute the actual posterior predictive for this;
# i.e. integrate over our uncertainty about model parameters.
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]
evaluate_multiclass_regression_with_beta_estimate(
    covariates_test,
    labels_test,
    beta_mean,
    link_for_category_probabilities=Link.CBM_PROBIT,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)
evaluate_sklearn_on_multiclass_regression(
    covariates_test,
    labels_test,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)


"""
SUMMARY OF RESULTS

    At some point,  we were beating sklearn on both training and test data, although I'm having
    trouble matching that now.
    
    Note: Possibly the "beating" simply reflects that the probit link is
    lighter tailed than the logit link.  It would be nice, though, if we could
    show that the phenomenon is caused by the properties of using a full posterior
    predictive, instead of a MAP estimate.

SOME SAMPLE RUNS:

When true generative process = CBC_PROBIT:

    # So here sklearn, but not the variational model,  is mis-specified

    Model check on training set...
    Now checking variational model with link function of Link.CBC_PROBIT
    The variational model's mean category probability is 0.715
    The sklearn model's mean category probability is 0.592


    Model check on test set...
    Now checking variational model with link function of Link.CBC_PROBIT
    The variational model's mean category probability is 0.727
    The sklearn model's mean category probability is 0.61

    # Note -in some situations sklearn, but not the var model, has a test set category prob which
    # drops enormously. 

When true generative process = MULTI_LOGIT :

    # So here, the variational model, but not sklearn, is mis-specified

    Model check on training set...
    Now checking variational model with link function of Link.CBC_PROBIT
    The variational model's mean category probability is 0.626
    The sklearn model's mean category probability is 0.582

    Now checking variational model with link function of Link.CBC_PROBIT
    The variational model's mean category probability is 0.632
    The sklearn model's mean category probability is 0.592

    # Note - I've seen far bigger differences - often .10 or even .20!
    # Unless there's a code error (or inappropriately used sklearn defaults), it 
    # suggests that the gains of VB outweight even MAP estimates, even though
    # the probit is mis-specified relative to the canonical multi-logit... 


Some info about defaults of scikit:

In [9]: lr.solver
Out[9]: 'lbfgs'

In [10]: lr.tol
Out[10]: 0.0001

In [12]: lr.multi_class
Out[12]: 'auto'
# docstring suggests we're using multinomial logit, not one vs all, 
# since the provided data is not binary: 
# "the loss minimised is the multinomial loss fit across the entire probability distribution"

In [20]: lr.penalty
Out[20]: 'l2'

In [21]: lr.fit_intercept
Out[21]: True

In [23]: lr.C
Out[23]: 1.0
# Inverse of regularization strength; must be a positive float. 
# Like in support vector machines, smaller values specify stronger regularization.
# perhaps one reason for VB's success is that the posterior sets the regularization strength
# (precision weighted mean) rather than having to preset it?

"""
