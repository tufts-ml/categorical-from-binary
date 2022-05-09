import numpy as np
import pytest

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_logit_with_DO_aux_var.inference import (
    compute_multiclass_logit_vi_with_CBC_aux_var_and_polya_gamma_augmentation,
)


@pytest.fixture
def dataset():
    n_categories = 3
    n_features = 3
    n_samples = 100
    include_intercept = True
    link = Link.MULTI_LOGIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
    beta_category_strategy = ControlCategoryPredictability(
        scale_for_predictive_categories=4.0
    )
    return generate_multiclass_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        beta_0=None,
        link=link,
        seed=None,
        include_intercept=include_intercept,
        beta_category_strategy=beta_category_strategy,
    )


def test__compute_multiclass_logit_vi_with_CBC_aux_var_and_polya_gamma_augmentation__gives_non_negligible_accuracy(
    dataset,
):

    # Prep training / test split
    n_samples, n_categories = np.shape(dataset.labels)
    n_train_samples = int(0.8 * n_samples)
    covariates_train = dataset.features[:n_train_samples]
    labels_train = dataset.labels[:n_train_samples]
    covariates_test = dataset.features[n_train_samples:]
    labels_test = dataset.labels[n_train_samples:]

    max_n_iterations = 3

    results = compute_multiclass_logit_vi_with_CBC_aux_var_and_polya_gamma_augmentation(
        labels_train,
        covariates_train,
        labels_test=labels_test,
        covariates_test=covariates_test,
        variational_params_init=None,
        max_n_iterations=max_n_iterations,
    )
    accuracy = results.performance_over_time["test accuracy with CBM_LOGIT"]
    minimum_acceptable_accuracy = 1.25 / (n_categories)
    assert np.mean(accuracy) > minimum_acceptable_accuracy
