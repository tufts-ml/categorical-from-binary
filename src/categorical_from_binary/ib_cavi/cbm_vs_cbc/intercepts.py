"""
Goal: We show two things:

1) In the intercepts-only case, the IB betas optimize the CBM model,
    but do not optimize the CBC model.  Moreover, by toggling the beta_0
    for the true known data generating process,  we can show that the warping gets
    worse as the true category probabilities leave the center of the probability simplex
    and move towards a vertex.
2) The MLE equation for the CBC model is correct.
"""

import jax.numpy as np


np.set_printoptions(precision=3, suppress=True)


import collections

import pandas as pd
import scipy

from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions,
    compute_CBM_Probit_predictions,
    compute_IB_Probit_predictions,
    compute_training_loss,
    optimize_beta_for_CBC_model,
    optimize_beta_for_CBM_model,
    optimize_beta_for_IB_model,
)
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)


loss_results_dict = collections.defaultdict(list)
n_simulations = 1


for seed in range(n_simulations):
    print(f"----- Now running simulation {seed} ---- ")
    ###
    # Construct dataset
    ###
    n_categories = 3
    n_features = 0
    n_samples = 5000
    beta_0 = np.array([3, 2, 1])
    include_intercept = True
    link = Link.CBC_LOGIT
    dataset = generate_multiclass_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        beta_0=beta_0,
        link=link,
        seed=seed,
        include_intercept=include_intercept,
    )

    beta_init = np.zeros((n_features + 1, n_categories))

    beta_star_IB_MLE = optimize_beta_for_IB_model(
        beta_init, dataset.features, dataset.labels
    )

    beta_star_CBC_MLE = optimize_beta_for_CBC_model(
        beta_init, dataset.features, dataset.labels
    )
    # betas are very different for CBC and IB,but that's not surprising since CBC isn't identifiable
    beta_star_CBM_MLE = optimize_beta_for_CBM_model(
        beta_init, dataset.features, dataset.labels
    )

    CBC_predictions_with_IB_MLE = compute_CBC_Probit_predictions(
        beta_star_IB_MLE, dataset.features
    )
    CBC_predictions_with_CBC_MLE = compute_CBC_Probit_predictions(
        beta_star_CBC_MLE, dataset.features
    )
    # predictions look a bit different!!!! but how different is different?!?!

    CBM_predictions_with_IB_MLE = compute_CBM_Probit_predictions(
        beta_star_IB_MLE, dataset.features
    )
    CBM_predictions_with_CBM_MLE = compute_CBM_Probit_predictions(
        beta_star_CBM_MLE, dataset.features
    )

    category_probability_function_CBC = compute_CBC_Probit_predictions
    loss_with_CBC_model_and_IB_MLE = compute_training_loss(
        beta_star_IB_MLE,
        dataset.features,
        dataset.labels,
        category_probability_function_CBC,
        is_truly_categorical=True,
    )
    loss_with_CBC_model_and_CBC_MLE = compute_training_loss(
        beta_star_CBC_MLE,
        dataset.features,
        dataset.labels,
        category_probability_function_CBC,
        is_truly_categorical=True,
    )

    category_probability_function_IB = compute_IB_Probit_predictions
    loss_with_IB_model_and_IB_MLE = compute_training_loss(
        beta_star_IB_MLE,
        dataset.features,
        dataset.labels,
        category_probability_function_IB,
        is_truly_categorical=False,
    )
    loss_with_IB_model_and_CBC_MLE = compute_training_loss(
        beta_star_CBC_MLE,
        dataset.features,
        dataset.labels,
        category_probability_function_IB,
        is_truly_categorical=False,
    )

    category_probability_function_CBM = compute_CBM_Probit_predictions
    loss_with_CBM_model_and_IB_MLE = compute_training_loss(
        beta_star_IB_MLE,
        dataset.features,
        dataset.labels,
        category_probability_function_CBM,
        is_truly_categorical=False,
    )
    loss_with_CBM_model_and_CBM_MLE = compute_training_loss(
        beta_star_CBM_MLE,
        dataset.features,
        dataset.labels,
        category_probability_function_CBM,
        is_truly_categorical=False,
    )

    loss_results_dict["CBC model, IB MLE"].append(loss_with_CBC_model_and_IB_MLE)
    loss_results_dict["CBC model, CBC MLE"].append(loss_with_CBC_model_and_CBC_MLE)
    loss_results_dict["IB model, IB MLE"].append(loss_with_IB_model_and_IB_MLE)
    loss_results_dict["IB model, CBC MLE"].append(loss_with_IB_model_and_CBC_MLE)
    loss_results_dict["CBM model, IB MLE"].append(loss_with_CBM_model_and_IB_MLE)
    loss_results_dict["CBM model, CBM MLE"].append(loss_with_CBM_model_and_CBM_MLE)

df = pd.DataFrame(loss_results_dict)
df
### Question of interest #1: is the loss about the same for the CBC model regardless of whether
# we plug in the IB model or the CBC model?  What even counts as about the same?
#
# Well, the additional loss from using the IB estimator seems much smaller for the CBM model
# than for the CBC model.
#
# More directly than this, check out the predictions ... it suggests that the point estimate for the IB model
# is NOT one of the MLE's for the CBC model!!! But the point estimate for the IB model IS one of the MLE's
# for the CBM model.

"""

# it's better when the cat probs are uniform. we can prove that in the intercepts-only case, 
# the MLE for IB is an MLE for CBC ONLY when the category probabilities are uniform!

CBM predictions with CBM MLE: [0.323 0.336 0.341]
CBM predictions with IB MLE: [0.324 0.336 0.34 ]
CBC predictions with CBC MLE: [0.323 0.336 0.341]
CBC predictions with IB MLE: [0.319 0.337 0.344]

# the more extreme the base rates, the worse off CBC is. 

CBM predictions with CBM MLE: [0.417 0.308 0.274]
CBM predictions with IB MLE: [0.417 0.308 0.274]
CBC predictions with CBC MLE: [0.417 0.308 0.274]
CBC predictions with IB MLE: [0.465 0.289 0.246]

CBM predictions with CBM MLE: [0.672 0.243 0.085]
CBM predictions with IB MLE: [0.671 0.243 0.085]
CBC predictions with CBC MLE: [0.672 0.243 0.085]
CBC predictions with IB MLE: [0.831 0.131 0.038]



"""
# So where was Johndrow's argument wrong? And can we adjust the argument to work for CBM?
#
print(f"CBM predictions with CBM MLE: {CBM_predictions_with_CBM_MLE[0]}")
print(f"CBM predictions with IB MLE: {CBM_predictions_with_IB_MLE[0]}")
print(f"CBC predictions with CBC MLE: {CBC_predictions_with_CBC_MLE[0]}")
print(f"CBC predictions with IB MLE: {CBC_predictions_with_IB_MLE[0]}")


## Question of interest #2: Can we verify the MLE formula for CBC? See Eqn 2.4.3 in workshop paper.
Phi = scipy.stats.norm.cdf
Phi_inv = scipy.stats.norm.ppf
Psi = lambda eta: Phi(eta) / Phi(-eta)

empirical_probs = np.mean(dataset.labels, 0)
s = np.sum(
    Psi(beta_star_CBC_MLE)
)  # infinitely many s's would work, but we pick the one from the estimate.
beta_CBC_MLE_analytical = Phi_inv(s * empirical_probs / (1 + s * empirical_probs))
assert np.isclose(beta_CBC_MLE_analytical, beta_star_CBC_MLE, atol=0.01).all()

# an alternate MLE for CBC
s = 0.1  # pick any s
beta_another_CBC_MLE_analytical = Phi_inv(
    s * empirical_probs / (1 + s * empirical_probs)
)[np.newaxis, :]
CBC_predictions_with_another_CBC_MLE = compute_CBC_Probit_predictions(
    beta_another_CBC_MLE_analytical, dataset.features
)
assert np.isclose(
    CBC_predictions_with_CBC_MLE, CBC_predictions_with_another_CBC_MLE, atol=0.01
).all()
