from typing import List, Optional, Union

import yaml
from pydantic import BaseModel

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.datasets.generic.load import Dataset


###
# Data Configs
###


class Generic_Real_Data_Configs(BaseModel):
    dataset: Dataset
    standardize_design_matrix: bool
    pct_training: float
    seed: int


class Cyber_Data_Configs(BaseModel):
    # The number of categories will be given by the number of unique categories launched across
    # the subset
    path_to_human_process_start_data: str
    subset_initial_user_idx_when_sorting_most_to_fewest_events: Optional[int] = None
    subset_number_of_users: Optional[int] = 1
    user_idx_relative_to_subset: Optional[int] = 0
    window_size: int
    temperature: float
    include_intercept: bool
    pct_training: float


class Simulated_Data_Configs(BaseModel):
    n_categories: int  # TODO: This should instead be accessed directly from the dataset object
    n_features: int
    n_samples: int
    seed: int
    include_intercept: bool
    link: Link
    scale_for_predictive_categories = 2.0
    pct_training: float


class Data_Configs(BaseModel):
    cyber: Optional[Cyber_Data_Configs] = None
    generic_real: Optional[Generic_Real_Data_Configs] = None
    sim: Optional[Simulated_Data_Configs] = None


###
# Holdout performance configs
###


class ADVI_Configs(BaseModel):
    n_iterations: int
    seed: int
    link: Link
    lrs: List[float]
    save_beta_every_secs: Optional[float] = None


class PGA_Softmax_Gibbs_Configs(BaseModel):
    n_samples: int
    pct_burn_in: float
    stride_for_evaluating_holdout_performance: int


class IB_CAVI_Probit_Configs(BaseModel):
    n_iterations: int
    save_beta_every_secs: Optional[float] = None


class IB_CAVI_Logit_Configs(BaseModel):
    n_iterations: int
    save_beta_every_secs: Optional[float] = None


class NUTS_Configs(BaseModel):
    n_warmup: int
    n_mcmc_samples: int
    stride_for_evaluating_holdout_performance: int
    link: Link
    seed: int


class Holdout_Performance_Over_Time_Configs(BaseModel):
    advi: Optional[ADVI_Configs] = None
    cavi_probit: Optional[IB_CAVI_Probit_Configs] = None
    cavi_logit: Optional[IB_CAVI_Logit_Configs] = None
    nuts: Optional[NUTS_Configs] = None
    pga_softmax_gibbs: Optional[PGA_Softmax_Gibbs_Configs] = None


###
# Meta configs
###


class Meta_Configs(BaseModel):
    purpose: str
    save_dir: Optional[str] = None


###
# Plotting configs
###


class Plotting_Configs(BaseModel):
    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve: float
    min_log_likelihood_for_y_axis: Optional[Union[str, float]] = None
    max_log_likelihood_for_y_axis: Optional[float] = None


###
# All configs
###


class Configs(BaseModel):
    meta: Meta_Configs
    data: Data_Configs
    holdout_performance: Holdout_Performance_Over_Time_Configs
    plot: Optional[Plotting_Configs] = None


###
# Loader
###


def load_configs(path_to_configs: str):
    with open(path_to_configs, "r") as f:
        return Configs(**yaml.full_load(f))
