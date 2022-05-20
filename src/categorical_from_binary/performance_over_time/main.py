import argparse
import os
from typing import Optional

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.datasets.cyber.load import (
    construct_process_start_features_and_labels_for_one_cyber_user,
)
from categorical_from_binary.datasets.generic.load import construct_data_split
from categorical_from_binary.io import ensure_dir, write_json
from categorical_from_binary.metrics import Metrics, compute_metrics
from categorical_from_binary.mst_time import get_mst_time
from categorical_from_binary.performance_over_time.classes import InferenceType
from categorical_from_binary.performance_over_time.configs_util import (
    Configs,
    load_configs,
)
from categorical_from_binary.performance_over_time.core import (
    compute_performance_over_time,
)
from categorical_from_binary.performance_over_time.metadata import MetaData
from categorical_from_binary.performance_over_time.plotter import (
    plot_performance_over_time_results,
)
from categorical_from_binary.performance_over_time.write import (
    write_performance_over_time_results,
)


def _get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run performance over time from path to yaml configs"
    )
    parser.add_argument(
        "--path_to_configs",
        type=str,
    )
    parser.add_argument(
        "--only_run_this_inference",
        type=int,
        help=f"integer representation of InferenceType Enum; if not specified we run all inference methods",
        default=None,
    )
    parser.add_argument(
        "--cyber_user_idx_relative_to_subset_override",
        type=int,
        help=f"Can be used to override user idx relative to subset when operating on cyber data",
        default=None,
    )
    parser.add_argument(
        "--make_plots",
        type=bool,
        help=f"whether to make plots; defaults to false because will cause code to raise an error if all inf methods not specified",
        default=False,
    )
    return parser


def run_performance_over_time_from_loaded_configs(
    configs: Configs,
    make_plots: bool = True,
) -> None:
    """
    Arguments:
        make_plots:  Can set to False if ` only_run_this_inference` is not None, because
            the plotter currently assumes that certain inference methods (e.g. ADVI) are present.

    Usage:

        path_to_configs = "configs/performance_over_time/demo_sims.yaml"
        configs = load_configs(path_to_configs)

    """

    ###
    # Construct dataset
    ###

    if configs.data.sim is not None:
        dataset = generate_multiclass_regression_dataset(
            n_samples=configs.data.sim.n_samples,
            n_features=configs.data.sim.n_features,
            n_categories=configs.data.sim.n_categories,
            beta_0=None,
            link=configs.data.sim.link,
            seed=configs.data.sim.seed,
            include_intercept=configs.data.sim.include_intercept,
            beta_category_strategy=ControlCategoryPredictability(
                configs.data.sim.scale_for_predictive_categories
            ),
        )

        # Prep training / test split
        n_train_samples = int(
            configs.data.sim.pct_training * configs.data.sim.n_samples
        )
        covariates_train = dataset.features[:n_train_samples]
        labels_train = dataset.labels[:n_train_samples]
        covariates_test = dataset.features[n_train_samples:]
        labels_test = dataset.labels[n_train_samples:]

    elif configs.data.generic_real is not None:

        split_dataset = construct_data_split(
            configs.data.generic_real.dataset,
            configs.data.generic_real.pct_training,
            configs.data.generic_real.standardize_design_matrix,
            configs.data.generic_real.seed,
        )
        labels_train = split_dataset.labels_train
        covariates_train = split_dataset.covariates_train
        labels_test = split_dataset.labels_test
        covariates_test = split_dataset.covariates_test

    elif configs.data.cyber is not None:
        (
            covariates,
            labels,
            user_domain,
        ) = construct_process_start_features_and_labels_for_one_cyber_user(
            configs.data.cyber.path_to_human_process_start_data,
            configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events,
            configs.data.cyber.subset_number_of_users,
            configs.data.cyber.user_idx_relative_to_subset,
            configs.data.cyber.window_size,
            configs.data.cyber.temperature,
            configs.data.cyber.include_intercept,
        )  # noqa

        # Prep training / test split
        n_train_samples = int(configs.data.cyber.pct_training * np.shape(covariates)[0])
        covariates_train = covariates[:n_train_samples]
        labels_train = labels[:n_train_samples]
        covariates_test = covariates[n_train_samples:]
        labels_test = labels[n_train_samples:]

    ###
    # Data subset
    ###
    data_subset = user_domain if configs.data.cyber is not None else ""

    ###
    # Metadata
    ###
    # metadata includes performance of anchors: random guessing and, if applicable, data generating process
    if configs.data.sim is not None:
        cat_probs_data_degenerating_process_test = construct_category_probs(
            covariates_test, dataset.beta, link=configs.data.sim.link
        )
        metrics_data_generating_process = compute_metrics(
            cat_probs_data_degenerating_process_test, labels_test
        )
        print(
            f"Mean holdout log like with true model is {metrics_data_generating_process.mean_log_like}"
        )
    else:
        metrics_data_generating_process = Metrics()

    n_categories = np.shape(labels_train)[1]
    metadata = MetaData(
        n_categories,
        mean_log_like_random_guessing=np.log(1.0 / n_categories),
        accuracy_random_guessing=1.0 / n_categories,
        mean_log_like_data_generating_process=metrics_data_generating_process.mean_log_like,
        accuracy_data_generating_process=metrics_data_generating_process.accuracy,
    )
    print(f"Metadata for this dataset: {metadata}")

    ###
    # Construct directory for saving results
    ###
    mst_time = get_mst_time()
    inference_used_list = [
        k for (k, v) in configs.holdout_performance.dict().items() if v is not None
    ]
    inference_used_string = "+".join(inference_used_list)
    save_dir_with_purpose_and_subset_and_time_and_inference_used = os.path.join(
        f"{configs.meta.save_dir}", data_subset, f"{mst_time}_{ inference_used_string}"
    )
    ensure_dir(save_dir_with_purpose_and_subset_and_time_and_inference_used)

    ###
    # Run holdout performance
    ###
    performance_over_time_results = compute_performance_over_time(
        covariates_train,
        labels_train,
        covariates_test,
        labels_test,
        configs.holdout_performance,
        save_dir_with_purpose_and_subset_and_time_and_inference_used,
    )

    ###
    # Save other artifacts
    ###

    # Configs
    write_json(
        configs.dict(),
        os.path.join(
            save_dir_with_purpose_and_subset_and_time_and_inference_used, "configs.json"
        ),
    )

    # Metadata
    write_json(
        metadata.__dict__,
        os.path.join(
            save_dir_with_purpose_and_subset_and_time_and_inference_used,
            "metadata.json",
        ),
    )

    # Dataframes
    write_performance_over_time_results(
        performance_over_time_results,
        os.path.join(
            save_dir_with_purpose_and_subset_and_time_and_inference_used,
            "result_data_frames",
        ),
    )

    # Plots
    if make_plots and configs.plot is not None:
        for show_cb_logit in [True, False]:
            for add_legend_to_plot in [True, False]:
                plot_performance_over_time_results(
                    performance_over_time_results,
                    os.path.join(
                        save_dir_with_purpose_and_subset_and_time_and_inference_used,
                        "plots",
                    ),
                    metadata.mean_log_like_data_generating_process,
                    metadata.accuracy_data_generating_process,
                    metadata.mean_log_like_random_guessing,
                    metadata.accuracy_random_guessing,
                    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve=configs.plot.min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve,
                    min_log_likelihood_for_y_axis=configs.plot.min_log_likelihood_for_y_axis,
                    add_legend_to_plot=add_legend_to_plot,
                    show_cb_logit=show_cb_logit,
                    nuts_link_name=configs.holdout_performance.nuts.link.name,
                )


def update_configs_via_optional_overrides(
    configs: Configs,
    only_run_this_inference: Optional[bool],
    cyber_user_idx_relative_to_subset_override: Optional[int],
) -> Configs:
    """
    Use case is setting overrides to yaml files in command line.
    These overrides can help with parallelization.

    Arguments:
        only_run_this_inference: Allow us to choose a single inference method to run;
            this allows us to run many inference methods on the same dataset in parallel
            (which is useful for larger runs)
    """
    # remove inference types we are not using
    if only_run_this_inference is not None:
        only_run_this_inference = InferenceType(only_run_this_inference)

        inference_string_by_inference_type_to_remove = {
            InferenceType.ADVI: "advi",
            InferenceType.CAVI_LOGIT: "cavi_logit",
            InferenceType.CAVI_PROBIT: "cavi_probit",
            InferenceType.NUTS: "nuts",
            InferenceType.SOFTMAX_VIA_PGA_AND_GIBBS: "pga_softmax_gibbs",
        }

        del inference_string_by_inference_type_to_remove[only_run_this_inference]
        for inference_string in inference_string_by_inference_type_to_remove.values():
            setattr(configs.holdout_performance, inference_string, None)

    if cyber_user_idx_relative_to_subset_override is not None:
        configs.data.cyber.user_idx_relative_to_subset = (
            cyber_user_idx_relative_to_subset_override
        )
        user_id = (
            configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events
            + configs.data.cyber.user_idx_relative_to_subset
        )

        # configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events = subset_initial_user_idx_when_sorting_most_to_fewest_events
        configs.meta.purpose = f"cyber_for_{configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events}_user_in_subset_from_{configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events}_to_{configs.data.cyber.subset_initial_user_idx_when_sorting_most_to_fewest_events+configs.data.cyber.subset_number_of_users}"
        configs.meta.save_dir = os.path.join(configs.meta.save_dir, f"user_{user_id}")

    return configs


def run_performance_over_time(
    path_to_configs: str,
    only_run_this_inference: Optional[InferenceType] = None,
    cyber_user_idx_relative_to_subset_override: Optional[int] = None,
    make_plots: bool = True,
) -> None:
    """
    Run performance over time from yaml configs, but allow some overrides.

    Arguments:
        only_run_this_inference: Allow us to choose a single inference method to run;
            this allows us to run many inference methods on the same dataset in parallel
            (which is useful for larger runs)
        make_plots:  Can set to False if ` only_run_this_inference` is not None, because
            the plotter currently assumes that certain inference methods (e.g. ADVI) are present.
    """
    if only_run_this_inference is not None and make_plots is True:
        raise NotImplementedError(
            "We cannot currently make plots if we're not running all the inference types."
        )
    configs = load_configs(path_to_configs)
    configs = update_configs_via_optional_overrides(
        configs, only_run_this_inference, cyber_user_idx_relative_to_subset_override
    )
    run_performance_over_time_from_loaded_configs(configs, make_plots)


if __name__ == "__main__":
    """
    Usage:
        Example:
            python src/categorical_from_binary/performance_over_time/main.py  --path_to_configs configs/performance_over_time/demo_sims.yaml --only_run_this_inference 1
    """

    args = _get_argument_parser().parse_args()
    run_performance_over_time(
        args.path_to_configs,
        args.only_run_this_inference,
        args.cyber_user_idx_relative_to_subset_override,
        args.make_plots,
    )
