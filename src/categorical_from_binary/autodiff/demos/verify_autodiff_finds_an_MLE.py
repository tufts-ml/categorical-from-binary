"""
Here we corroborate that the autodiff is indeed finding (an) MLE from the set
of possible MLE's for the CBC model.  

EXPERIMENT 1: We sample random initializations from a Gaussian, and verify that we obtain
virtually identical training losses and category probabilities. 

EXPERIMENT 2 (older): We initialize the optimization
from two different values (zero matrix and data generating beta), and show that we get the same results (in terms of 
category probabilities, even if not in terms of beta.)

"""

from dataclasses import dataclass

import numpy as np


np.set_printoptions(precision=3, suppress=True)

from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions,
    compute_CBM_Probit_predictions,
    optimize_beta_and_return_beta_star_and_loss,
    optimize_beta_for_CBC_model,
)
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)


@dataclass
class DataGenerationConfigs:
    seed: int
    n_categories: int
    n_features: int
    n_samples: int
    scale_for_predictive_categories: float


beta_0 = None
include_intercept = True
link = Link.CBC_PROBIT
# data generation configs obtained from experiments.approximation_error.main.make_plots_of_signed_approxiamtion_error
data_generation_configs = [
    DataGenerationConfigs(4, 3, 1, 1800, 0.1),
    DataGenerationConfigs(6, 20, 40, 12000, 0.1),
]
dgc = data_generation_configs[0]

###
# Construct dataset
###
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=dgc.scale_for_predictive_categories
)
dataset = generate_multiclass_regression_dataset(
    n_samples=dgc.n_samples,
    n_features=dgc.n_features,
    n_categories=dgc.n_categories,
    beta_0=beta_0,
    link=link,
    seed=dgc.seed,
    include_intercept=include_intercept,
    beta_category_strategy=beta_category_strategy,
)


################
# EXPERIMENT 1 #
################
beta_init = np.zeros((dgc.n_features + 1, dgc.n_categories))
n_random_inits = 5

# CBC-Probit Experiment
print(f"\n---- Now running CBC-Probit experiment with {n_random_inits} inits. ----\n")
for i in range(n_random_inits):
    print("New initialization.\n")
    beta_init_random = np.random.normal(loc=0.0, scale=1.0, size=np.shape(beta_init))
    beta_star, loss = optimize_beta_and_return_beta_star_and_loss(
        beta_init_random,
        dataset.features,
        dataset.labels,
        category_probability_function=compute_CBC_Probit_predictions,
        is_truly_categorical=True,
        verbose=True,
    )
    print(f"Beta star:{beta_star}.\n\n")

# CBM-Probit Experiment
print(f"\n---- Now running CBM-Probit experiment with {n_random_inits} inits. ----\n")
for i in range(n_random_inits):
    print("New initialization.\n")
    beta_init_random = np.random.normal(loc=0.0, scale=1.0, size=np.shape(beta_init))
    beta_star, loss = optimize_beta_and_return_beta_star_and_loss(
        beta_init_random,
        dataset.features,
        dataset.labels,
        category_probability_function=compute_CBM_Probit_predictions,
        is_truly_categorical=True,
        verbose=True,
    )
    print(f"Beta star:{beta_star}.\n\n")

########################
# EXPERIMENT 2 (older) #
########################

###
# Compute MLE's
###

beta_init = np.zeros((dgc.n_features + 1, dgc.n_categories))

print("Now doing MLE with CBC model initializing at 0")
beta_star_CBC_MLE_when_initialized_at_zero = optimize_beta_for_CBC_model(
    beta_init, dataset.features, dataset.labels
)

print("Now doing MLE with CBC model initializating at the data generating value")
beta_star_CBC_MLE_when_initialized_at_data_generating_value = (
    optimize_beta_for_CBC_model(dataset.beta, dataset.features, dataset.labels)
)

# ` beta_star_CBC_MLE_when_initialized_at_zero` and `beta_star_CBC_MLE_when_initialized_at_data_generating_value`
# may be differnet (because the model is non-identifiable) even if they map to the same probability
# distribution over categories.  So we check the error in the cateogry probabilities, not in the betas.

###
# Compute difference in category probabilities
###

CBC_predictions_with_CBC_MLE_when_initialized_at_zero = compute_CBC_Probit_predictions(
    beta_star_CBC_MLE_when_initialized_at_zero, dataset.features, return_numpy=True
)

CBC_predictions_with_CBC_MLE_when_initialized_at_the_data_generating_value = (
    compute_CBC_Probit_predictions(
        beta_star_CBC_MLE_when_initialized_at_data_generating_value,
        dataset.features,
        return_numpy=True,
    )
)

mean_absolute_difference_in_category_probabilities = np.mean(
    np.abs(
        CBC_predictions_with_CBC_MLE_when_initialized_at_zero
        - CBC_predictions_with_CBC_MLE_when_initialized_at_the_data_generating_value
    )
)
print(
    f"The mean absolute difference in category probabilities from using the two different initializations schemes was {mean_absolute_difference_in_category_probabilities}"
)
