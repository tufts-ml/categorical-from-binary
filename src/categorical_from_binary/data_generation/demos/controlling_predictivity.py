import numpy as np


np.set_printoptions(precision=3, suppress=True)

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    compute_covariate_conditional_entropies_of_true_category_probabilities,
    generate_multiclass_regression_dataset,
)


n_categories = 50
n_sparse_categories = 0
n_features = 20
n_sparse_features = 0
n_samples = 5000
n_train_samples = 4000
include_intercept = True
link_for_data_generation = Link.MULTI_LOGIT
scale_for_intercept = 0.01
seed = 1

beta_category_strategies = [
    ControlCategoryPredictability(scale_for_predictive_categories=0.001),
    ControlCategoryPredictability(scale_for_predictive_categories=0.5),
    ControlCategoryPredictability(scale_for_predictive_categories=1.0),
    ControlCategoryPredictability(scale_for_predictive_categories=1.5),
    ControlCategoryPredictability(scale_for_predictive_categories=2.0),
]
for beta_category_strategy in beta_category_strategies:

    dataset = generate_multiclass_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        n_categories_where_all_beta_coefficients_are_sparse=n_sparse_categories,
        n_sparse_features=n_sparse_features,
        beta_0=None,
        link=link_for_data_generation,
        seed=seed,
        include_intercept=include_intercept,
        beta_category_strategy=beta_category_strategy,
        scale_for_intercept=scale_for_intercept,
    )
    np.set_printoptions(edgeitems=4)

    # probs_dgp=construct_category_probs(dataset.features, dataset.beta, link_for_data_generation)
    # mean_category_useage_unsorted = np.mean(probs_dgp,0)
    # mean_category_useage_sorted = np.sort(np.mean(probs_dgp,0))

    # print("\nMean category useage (sorted)")
    # print(mean_category_useage_sorted)

    # print("\nMean category useage (unsorted)")
    # print(mean_category_useage_unsorted)

    # print("\nSome example probabilities")
    # print(probs_dgp)

    # # predictivity analysis
    # mean_entropies_of_true_category_probabilities_by_label=compute_mean_entropies_of_true_category_probabilities_by_label(dataset.features, dataset.labels, dataset.beta,link.MULTI_LOGIT)
    # print(f"Mean entropies of true category probabilities by label: {mean_entropies_of_true_category_probabilities_by_label}")
    entropies_of_true_category_probabilities = (
        compute_covariate_conditional_entropies_of_true_category_probabilities(
            dataset.features, dataset.beta, link_for_data_generation
        )
    )
    mean_entropy_of_true_category_probabilities = np.mean(
        entropies_of_true_category_probabilities
    )
    print(
        f"\nWith {beta_category_strategy}, the mean entropy of true category probs was: {mean_entropy_of_true_category_probabilities :.04}"
    )
