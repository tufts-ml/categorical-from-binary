import pytest

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_logistic_regression_dataset,
)


@pytest.fixture()
def logistic_regression_dataset(n_samples=1000, n_features=5):
    return generate_logistic_regression_dataset(n_samples, n_features)
