from typing import Optional

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.hmc.core import (
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_ib_cavi_with_normal_prior,
)
from categorical_from_binary.kucukelbir.inference import (
    ADVI_Results,
    Metadata,
    do_advi_inference_via_kucukelbir_algo,
)
from categorical_from_binary.performance_over_time.classes import (
    InferenceType,
    PerformanceOverTimeResults,
)
from categorical_from_binary.performance_over_time.configs_util import (
    Holdout_Performance_Over_Time_Configs,
)
from categorical_from_binary.performance_over_time.for_mcmc import (
    construct_performance_over_time_for_MCMC,
)
from categorical_from_binary.polya_gamma.multiclass_logreg_gibbs.inference import (
    sample_from_posterior_of_multiclass_logistic_regression_with_pga,
)
from categorical_from_binary.timing import time_me
from categorical_from_binary.types import NumpyArray2D


def compute_performance_over_time(
    covariates_train: NumpyArray2D,
    labels_train: NumpyArray2D,
    covariates_test: NumpyArray2D,
    labels_test: NumpyArray2D,
    configs: Holdout_Performance_Over_Time_Configs,
    save_beta_dir: Optional[str] = None,
    only_run_this_inference: Optional[InferenceType] = None,
) -> PerformanceOverTimeResults:

    """
    If configs are set to None, then we don't run that thing.

    Arguments:
        data_subset: e.g., user_domain for cyber.  We want this info when we save stuff.
    """

    # Initialization (some of these methods may not be run)
    advi_results_by_lr = None
    performance_cavi_probit = None
    performance_cavi_logit = None
    performance_nuts = None
    performance_softmax_via_pga_and_gibbs = None

    ###
    # ADVI
    ###
    if configs.advi is not None and (
        only_run_this_inference is None or only_run_this_inference == InferenceType.ADVI
    ):

        N, M = np.shape(covariates_train)
        K = np.shape(labels_train)[1]
        need_to_add_intercept = False
        metadata = Metadata(N, M, K, need_to_add_intercept)

        advi_results_by_lr = {}
        for lr in configs.advi.lrs:
            print(f"\r--- Now doing ADVI with lr {lr:.05f} ---")
            (
                beta_mean_ADVI,
                beta_stds_ADVI,
                performance_ADVI,
            ) = do_advi_inference_via_kucukelbir_algo(
                labels_train,
                covariates_train,
                metadata,
                configs.advi.link,
                configs.advi.n_iterations,
                lr,
                configs.advi.seed,
                labels_test=labels_test,
                covariates_test=covariates_test,
                save_beta_every_secs=configs.advi.save_beta_every_secs,
                save_beta_dir=save_beta_dir,
            )
            ADVI_results = ADVI_Results(
                beta_mean_ADVI, beta_stds_ADVI, performance_ADVI
            )
            print(f"\n Performance with ADVI (lr={lr}): {performance_ADVI}")
            advi_results_by_lr[lr] = ADVI_results

    ####
    # Variational Inference with multiclass probit regression
    ####
    if configs.cavi_probit is not None and (
        only_run_this_inference is None
        or only_run_this_inference == InferenceType.CAVI_PROBIT
    ):

        results_CAVI_probit = compute_ib_cavi_with_normal_prior(
            IB_Model.PROBIT,
            labels_train,
            covariates_train,
            labels_test=labels_test,
            covariates_test=covariates_test,
            variational_params_init=None,
            # convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
            max_n_iterations=configs.cavi_probit.n_iterations,
            save_beta_every_secs=configs.cavi_probit.save_beta_every_secs,
            save_beta_dir=save_beta_dir,
        )
        performance_cavi_probit = results_CAVI_probit.performance_over_time
        print(f"\n Performance with CAVI probit: {performance_cavi_probit}")

    ####
    # Variational Inference with multiclass logit regression
    ####
    if configs.cavi_logit is not None and (
        only_run_this_inference is None
        or only_run_this_inference == InferenceType.CAVI_LOGIT
    ):

        results_CAVI_logit = compute_ib_cavi_with_normal_prior(
            IB_Model.LOGIT,
            labels_train,
            covariates_train,
            labels_test=labels_test,
            covariates_test=covariates_test,
            variational_params_init=None,
            # convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
            max_n_iterations=configs.cavi_logit.n_iterations,
            save_beta_every_secs=configs.cavi_logit.save_beta_every_secs,
            save_beta_dir=save_beta_dir,
        )
        performance_cavi_logit = results_CAVI_logit.performance_over_time
        print(f"\n Performance with CAVI probit: {performance_cavi_logit}")

    ####
    # NUTS
    ####
    if configs.nuts is not None and (
        only_run_this_inference is None or only_run_this_inference == InferenceType.NUTS
    ):
        n_train_samples = np.shape(labels_train)[0]
        Nseen_list = [n_train_samples]
        beta_samples_NUTS_dict, time_for_NUTS = time_me(run_nuts_on_categorical_data)(
            configs.nuts.n_warmup,
            configs.nuts.n_mcmc_samples,
            Nseen_list,
            create_categorical_model,
            configs.nuts.link,
            labels_train,
            covariates_train,
            configs.nuts.seed,
        )

        beta_samples_NUTS = np.array(beta_samples_NUTS_dict[n_train_samples])
        performance_nuts = construct_performance_over_time_for_MCMC(
            beta_samples_NUTS,
            time_for_NUTS,
            covariates_train,
            labels_train,
            covariates_test,
            labels_test,
            configs.nuts.link,
            stride=configs.nuts.stride_for_evaluating_holdout_performance,
            n_warmup_samples=configs.nuts.n_warmup,
            one_beta_sample_has_transposed_orientation=True,
        )
        print(f"\n\nNUTS holdout performance over time: \n {performance_nuts }")

    ###
    # Gibbs Softmax with PGA
    ###

    if configs.pga_softmax_gibbs is not None and (
        only_run_this_inference is None
        or only_run_this_inference == InferenceType.SOFTMAX_VIA_PGA_AND_GIBBS
    ):

        ## inference
        beta_samples_gibbs_with_warmup, time_for_gibbs_with_warmup = time_me(
            sample_from_posterior_of_multiclass_logistic_regression_with_pga
        )(
            covariates_train,
            labels_train,
            configs.pga_softmax_gibbs.n_samples,
            prior_info=None,
            beta_init=None,
        )

        num_burn_in = int(
            configs.pga_softmax_gibbs.pct_burn_in * configs.pga_softmax_gibbs.n_samples
        )
        beta_samples_gibbs_without_warmup = beta_samples_gibbs_with_warmup[
            num_burn_in:, :, :
        ]
        # beta_posterior_mean_from_gibbs= np.mean(beta_samples[num_burn_in:], 0)
        performance_softmax_via_pga_and_gibbs = construct_performance_over_time_for_MCMC(
            beta_samples_gibbs_without_warmup,
            time_for_gibbs_with_warmup,
            covariates_train,
            labels_train,
            covariates_test,
            labels_test,
            Link.SOFTMAX,
            stride=configs.pga_softmax_gibbs.stride_for_evaluating_holdout_performance,
            n_warmup_samples=num_burn_in,
        )
        print(
            f"\n\nGibbs holdout performance over time: \n {performance_softmax_via_pga_and_gibbs}"
        )

    return PerformanceOverTimeResults(
        advi_results_by_lr,
        performance_cavi_probit,
        performance_cavi_logit,
        performance_nuts,
        performance_softmax_via_pga_and_gibbs,
    )
