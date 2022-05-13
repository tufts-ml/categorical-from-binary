import numpy as np

from categorical_from_binary.hmc.core import (
    CategoricalModelType,
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.hmc.generate import (
    generate_intercepts_only_categorical_data,
)


def test_hmc___demo_intercepts_only():
    """
    We just test that the demo function runs without error.
    """

    # TODO: Break this up into proper unit tests
    # TODO: Find a way to run this test without the high start-up costs of leading in HMC.

    # Configs
    random_seed = 42
    num_samples = 10
    true_category_probs_K = np.asarray([0.05, 0.95])

    # Generate (intercepts-only) categorical data
    (
        y_train__one_hot_NK,
        y_test__one_hot_NK,
    ) = generate_intercepts_only_categorical_data(
        true_category_probs_K,
        num_samples,
        random_seed=random_seed,
    )

    num_warmup, num_mcmc_samples = 0, 10
    Nseen_list = [10]
    categorical_model_type = CategoricalModelType.SOFTMAX
    betas_SLM_by_N = run_nuts_on_categorical_data(
        num_warmup,
        num_mcmc_samples,
        Nseen_list,
        create_categorical_model,
        categorical_model_type,
        y_train__one_hot_NK,
        random_seed=random_seed,
    )
    assert betas_SLM_by_N is not None
