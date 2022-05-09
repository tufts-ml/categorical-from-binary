from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


# TODO: This whole module triggers annoying logistics downstream which could be
# avoided if I just worked with dataframes instead of two separate objects, covariates
# and covariate dicts


class VariableType(Enum):
    BINARY = 1
    CONTINUOUS = 2


@dataclass
class VariableInfo:
    type: VariableType
    index: int
    raw_mean: Optional[float] = None
    raw_std: Optional[float] = None
    raw_min: Optional[float] = None
    raw_max: Optional[float] = None


def sorted_covariate_names_from_covariate_dict(
    covariate_dict: Dict[str, VariableInfo]
) -> List[str]:
    max_index = max(
        [covariate_info.index for covariate_info in covariate_dict.values()]
    )
    sorted_covariate_names = []
    for i in range(0, max_index + 1):
        for covariate_name, covariate_info in covariate_dict.items():
            if covariate_info.index == i:
                sorted_covariate_names.append(covariate_name)
    return sorted_covariate_names
