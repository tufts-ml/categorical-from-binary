"""
Here we corroborate that the autodiff is indeed finding (an) MLE from the set
of possible MLE's for the CBC model.  We do this by initializing the optimization
from two different values, and show that we get the same results (in terms of 
category probabilities, even if not in terms of beta.)
"""

from dataclasses import dataclass

import numpy as np


np.set_printoptions(precision=3, suppress=True)


from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions,
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
dgc = DataGenerationConfigs(4, 3, 1, 1800, 0.1)


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
