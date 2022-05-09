from dataclasses import dataclass
from typing import List


@dataclass
class DataGenerationConfig:
    """
    Note:
        `scale_for_predictive_categories` is a STANDARD DEVIATION (not a variance)
        It is the square root of what some modules represent as ssq_high.
    """

    seed: int
    n_categories: int
    n_features: int
    n_samples: int
    scale_for_predictive_categories: float


def get_num_parameters_from_num_covariates_and_num_categories(
    num_categories: int,
    num_covariates: int,
):
    K, M = num_categories, num_covariates
    return K * (M + 1)


def make_data_generation_configs(
    list_of_n_categories: List[int],
    multipliers_on_n_categories_to_create_n_covariates: List[int],
    multipliers_on_n_parameters_to_create_n_samples: List[int],
    list_of_scales_for_predictive_categories: List[float],
    seed: int = 1,
) -> List[DataGenerationConfig]:
    """
    Uses the provided arguments to create the info needed to generate categorical regression
    datasets; namey, (K,M,N,ssq_high), where K is the number of categories, M is the number of covariates,
    N is the number of samples, and ssq_high controls the categorical response predictability.

    Then takes the Cartesian product of these things to create a list of Data Generation Configs.
    """
    data_generation_configs = []
    for K in list_of_n_categories:
        for (
            multiplier_to_create_n_covariates
        ) in multipliers_on_n_categories_to_create_n_covariates:
            M = K * multiplier_to_create_n_covariates
            P = get_num_parameters_from_num_covariates_and_num_categories(K, M)
            for (
                multiplier_to_create_n_samples
            ) in multipliers_on_n_parameters_to_create_n_samples:
                N = P * multiplier_to_create_n_samples
                for scale in list_of_scales_for_predictive_categories:
                    data_generation_config = DataGenerationConfig(seed, K, M, N, scale)
                    data_generation_configs.append(data_generation_config)
    return data_generation_configs
