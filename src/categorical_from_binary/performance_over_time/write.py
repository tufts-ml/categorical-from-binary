import os
from typing import Dict, Optional

from pandas.core.frame import DataFrame

from categorical_from_binary.io import ensure_dir
from categorical_from_binary.kucukelbir.inference import ADVI_Results
from categorical_from_binary.performance_over_time.classes import PerformanceOverTimeResults


def write_performance_over_time_for_various_methods(
    save_dir: str,
    performance_cavi_probit: Optional[DataFrame] = None,
    performance_cavi_logit: Optional[DataFrame] = None,
    performance_softmax_via_pga_and_gibbs: Optional[DataFrame] = None,
    performance_nuts: Optional[DataFrame] = None,
    advi_results_by_lr: Optional[Dict[float, ADVI_Results]] = None,
) -> None:

    ensure_dir(save_dir)

    if performance_softmax_via_pga_and_gibbs is not None:
        fp = os.path.join(save_dir, "perf_softmax_via_pga_and_gibbs.csv")
        performance_softmax_via_pga_and_gibbs.to_csv(fp)

    if performance_nuts is not None:
        fp = os.path.join(save_dir, "perf_nuts.csv")
        performance_nuts.to_csv(fp)

    if performance_cavi_probit is not None:
        fp = os.path.join(save_dir, "perf_cavi_probit.csv")
        performance_cavi_probit.to_csv(fp)

    if performance_cavi_logit is not None:
        fp = os.path.join(save_dir, "perf_cavi_logit.csv")
        performance_cavi_logit.to_csv(fp)

    if advi_results_by_lr is not None:
        for lr in advi_results_by_lr.keys():
            fp = os.path.join(save_dir, f"perf_advi_{lr}.csv")
            advi_results_by_lr[lr].performance_ADVI.to_csv(fp)


def write_performance_over_time_results(
    performance_over_time_results: PerformanceOverTimeResults,
    save_dir: str,
):
    return write_performance_over_time_for_various_methods(
        save_dir,
        performance_over_time_results.df_performance_cavi_probit,
        performance_over_time_results.df_performance_cavi_logit,
        performance_over_time_results.df_performance_softmax_via_pga_and_gibbs,
        performance_over_time_results.df_performance_nuts,
        performance_over_time_results.advi_results_by_lr,
    )
