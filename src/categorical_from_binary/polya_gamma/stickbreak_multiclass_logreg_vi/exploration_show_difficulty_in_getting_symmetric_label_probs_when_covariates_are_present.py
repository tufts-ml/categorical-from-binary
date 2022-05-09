"""
Here we show that correcting the intercept term is insufficient to get symmetric category probabilities
when there are features present, even if the regression weights on those features are generated from a normal
distribution with mean zero in a way that is independent across categories.

When we have anything other than a random intercepts model, the labels favor earlier categories over later
categories. 
"""

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
    make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric,
)


n_datasets = 50
n_categories = 10
n_features_to_try = [0, 10]

label_proportions_per_dataset = np.zeros((n_datasets, n_categories))
for n_features in n_features_to_try:
    print(f"\n When the number of features is {n_features}...")
    for i in range(n_datasets):

        dataset = generate_multiclass_regression_dataset(
            n_samples=1000,
            n_features=n_features,
            n_categories=n_categories,
            beta_0=make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric(
                n_categories
            ),
            link=Link.STICK_BREAKING,
            seed=None,
        )
        label_proportions = np.mean(dataset.labels, 0)
        # print(f"The label proportions are {label_proportions}")
        label_proportions_per_dataset[i, :] = label_proportions

    mean_of_label_proportions_across_datasets = np.mean(
        label_proportions_per_dataset, 0
    )
    print(
        f"The mean of label proportions over datasets is {mean_of_label_proportions_across_datasets} "
    )


"""
Results on a typical run:

When the number of features is 0...
The mean of label proportions over datasets is [0.09868 0.0982  0.10122 0.10022 0.09888 0.10158 0.1003  0.09876 0.09998
 0.10218] 

When the number of features is 10...
The mean of label proportions over datasets is [0.2692  0.21408 0.15128 0.106   0.08002 0.06248 0.04324 0.0304  0.0226
 0.0207 ] 
"""
