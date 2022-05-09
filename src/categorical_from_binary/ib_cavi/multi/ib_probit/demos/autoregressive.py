"""
We demo an autoregressive categorical model which has closed-form
coordinate ascent variational inference (CAVI) updates.

The autoregressive model is the CBC model
with a particular choice of design matrix. 

For now, we just generate data with no autoregressive struture, and we show
that the model makes sensible inferences. 
"""

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


###
# Construct dataset with no autoregressive structure
###

n_categories = 3
n_features = 3
n_samples = 5000
link = Link.MULTI_LOGIT  # Link.CBC_PROBIT
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link,
    seed=None,
)


# Prep training data
n_train_samples = 4000
features_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]


###
# Fit autoregressive model
###

results_from_AR_model = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    features_train,
    variational_params_init=None,
    max_n_iterations=30,
    use_autoregressive_design_matrix=True,
)
variational_params_from_AR_model = results_from_AR_model.variational_params
beta_mean_from_AR_model, beta_cov_from_AR_model = (
    variational_params_from_AR_model.beta.mean,
    variational_params_from_AR_model.beta.cov,
)


# autoregressive version:
# mean category prob = 0.57

# Rk: At first, I expected to see the betas associated to prev lables (the first 3)
# be all 0's, since the prev labels have no effect here.  But then I remembered that
# the label-specific betas are replacing the intercept (decomposing it, in fact, into
# a preceding-label specific intercept.)  Thus, what we see instead is that
#   (a) they all hover around the same value (i.e. first three rows in each column are about the same)
#   (b) and in particualr, the hover around the value you get for the 1st row when runnign the non-AR version.
#
# It's interesting / cool that the other values of beta are (therefore?) not affected by the choice
# of AR vs. non-AR model...

"""

In [52]: beta_mean
Out[52]: 
array([[-0.53161303, -0.7662957 , -0.30897063],
       [-0.72984854, -0.62388489, -0.27093897],
       [-0.52395842, -0.78855578, -0.27809319],
       [ 0.94241119, -0.03055531, -0.80796653],
       [ 0.59212591,  0.04733466, -0.59194118],
       [-0.09873759, -0.04592223,  0.13370362]])
"""
###
# Standard model
###

results_from_standard_model = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    features_train,
    variational_params_init=None,
    n_iterations=30,
    use_autoregressive_design_matrix=False,
)

# standard version:
# mean category prob = 0.57 .... nice!

"""
In [58]: beta_mean
Out[58]: 
array([[-0.573327  , -0.7419813 , -0.28732169],
       [ 0.94157852, -0.03175925, -0.80775604],
       [ 0.59264368,  0.04610324, -0.5917188 ],
       [-0.10053772, -0.04517136,  0.13391043]])
"""
