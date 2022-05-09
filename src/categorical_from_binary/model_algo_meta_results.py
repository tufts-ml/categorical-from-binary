from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pandas.core.frame import DataFrame


# **** This is a WIP.  We're not currently using this. ####


class Model(Enum):
    TRUE_MODEL = 1
    SOFTMAX = 2
    CB_PROBIT = 3
    CB_LOGIT = 4


class Algo(Enum):
    NOT_APPLICABLE = 1
    IB_CAVI = 1
    HMC = 2
    ADVI = 3
    PGA_GIBBS = 4


@dataclass
class ModelAlgoMetaResults:
    model: Model
    algo: Algo
    meta_str: Optional[str] = None
    results: List[DataFrame] = field(default_factory=list)


def str_from_mamr(mamr: ModelAlgoMetaResults):
    return f"{mamr.model.name}+{mamr.algo.name}+{mamr.meta_str}"
