from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pandas.core.frame import DataFrame

from categorical_from_binary.kucukelbir.inference import ADVI_Results


# *** Mixing str and Enum as type allows an enum to be a dict key, while
# still syntactically behaving like an Enum.  Very nice!!!
# Reference: https://stackoverflow.com/questions/45459026/python-mix-in-enumerations-as-dictionary-key-how-the-type-is-converted
class InferenceType(int, Enum):
    CAVI_PROBIT = 1
    CAVI_LOGIT = 2
    NUTS = 3
    SOFTMAX_VIA_PGA_AND_GIBBS = 4
    ADVI = 5


@dataclass
class PerformanceOverTimeResults:
    # TODO: Check if I can rewritten functions that operate on this
    # (` plot_performance_over_time` and ` write_holdout_performance_over_time_for_various_methods`
    # to see if I can make all of these optional (or at least ADVI optional?)

    advi_results_by_lr: Dict[float, ADVI_Results]
    df_performance_cavi_probit: DataFrame
    df_performance_cavi_logit: Optional[DataFrame] = (None,)
    df_performance_nuts: Optional[DataFrame] = (None,)
    df_performance_softmax_via_pga_and_gibbs: Optional[DataFrame] = (None,)
