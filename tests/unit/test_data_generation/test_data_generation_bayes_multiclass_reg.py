import numpy as np
import pytest

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_multi_logit_probabilities,
    construct_stickbreaking_multinomial_probabilities,
    generate_designed_features_and_multiclass_labels_for_autoregressive_case,
    generate_multiclass_labels_for_nonautoregressive_case,
    generate_multiclass_regression_dataset,
    get_num_beta_columns,
    make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric,
)


def test_generate_multiclass_regression_dataset():
    """
    We test that we can run the function for various link functions and get the right
    number of categories in the responses, even though the beta dimensionality differs
    from link function to link function
    """
    n_categories, n_features, n_samples = 4, 5, 10
    include_intercept = False
    for link in [
        Link.CBC_PROBIT,
        Link.MULTI_LOGIT,
        Link.STICK_BREAKING,
        Link.CBC_LOGIT,
    ]:
        dataset = generate_multiclass_regression_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            beta_0=None,
            link=link,
            seed=None,
            include_intercept=include_intercept,
        )
        assert np.shape(dataset.labels)[1] == n_categories


@pytest.fixture
def multiclass_logistic_regression_dataset():
    return generate_multiclass_regression_dataset(
        n_samples=10, n_features=6, n_categories=3
    )


def test_that_the_construct_multi_logit_probabilities_function_returns_object_whose_rows_sum_to_unity(
    multiclass_logistic_regression_dataset,
):
    dataset = multiclass_logistic_regression_dataset
    probs = construct_multi_logit_probabilities(dataset.features, dataset.beta)
    assert (np.sum(probs, 1) == 1.0).all()


def test_that_the_construct_stickbreaking_multinomial_probabilities_function_returns_object_whose_rows_sum_to_unity(
    multiclass_logistic_regression_dataset,
):
    dataset = multiclass_logistic_regression_dataset
    probs = construct_stickbreaking_multinomial_probabilities(
        dataset.features, dataset.beta
    )
    assert (np.sum(probs, 1) == 1.0).all()


def test_make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric():
    # we make a regression where this is a single sample and where the only feature is an intercept term.
    n_samples, n_categories = 1, 4
    features = np.ones((n_samples, 1))
    beta = make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric(
        n_categories
    )[
        np.newaxis, :
    ]
    probs = construct_stickbreaking_multinomial_probabilities(features, beta)
    expected_category_probs = np.ones(n_categories) / n_categories
    assert np.isclose(probs, expected_category_probs).all()


def test_generate_multiclass_regression_dataset_with_stick_breaking_link():
    """
    We test that the function has the correct label proportions when there are no features
    """
    n_categories = 4
    dataset = generate_multiclass_regression_dataset(
        n_samples=10000,
        n_features=0,
        n_categories=n_categories,
        beta_0=make_stick_breaking_multinomial_regression_intercepts_such_that_category_probabilities_are_symmetric(
            n_categories
        ),
        link=Link.STICK_BREAKING,
        seed=2,
    )
    label_proportions = np.mean(dataset.labels, 0)
    expected_label_proportions = np.ones(n_categories) / n_categories
    assert np.isclose(label_proportions, expected_label_proportions, atol=0.03).all()


def test_generate_multiclass_labels_for_nonautoregressive_case():
    num_features_nonautoregressive = 5
    num_categories = 3
    num_samples = 100
    features_non_autoregressive = np.random.normal(
        loc=0, scale=1, size=(num_samples, num_features_nonautoregressive)
    )

    for link in Link:
        num_beta_columns = get_num_beta_columns(link, num_categories)
        beta = np.random.normal(
            loc=0, scale=1, size=(num_features_nonautoregressive, num_beta_columns)
        )
        labels = generate_multiclass_labels_for_nonautoregressive_case(
            features_non_autoregressive,
            beta,
            link,
        )
        assert labels is not None


def test_generate_designed_features_and_multiclass_labels_for_autoregressive_case():
    num_features_nonautoregressive = 5
    num_categories = 3
    num_samples = 100
    features_non_autoregressive = np.random.normal(
        loc=0, scale=1, size=(num_samples, num_features_nonautoregressive)
    )

    beta = np.random.normal(
        loc=0,
        scale=1,
        size=(num_features_nonautoregressive + num_categories, num_categories),
    )

    (
        designed_features,
        labels,
    ) = generate_designed_features_and_multiclass_labels_for_autoregressive_case(
        features_non_autoregressive,
        beta,
    )
    assert labels is not None
    assert np.shape(designed_features)[1] > np.shape(features_non_autoregressive)[1]
