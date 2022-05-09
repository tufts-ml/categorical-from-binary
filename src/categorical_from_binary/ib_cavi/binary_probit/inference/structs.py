from dataclasses import dataclass
from enum import Enum
from typing import Optional

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


@dataclass
class VariationalBeta:
    mean: NumpyArray1D
    cov: NumpyArray2D


@dataclass
class VariationalZs:
    parent_mean: NumpyArray1D


@dataclass
class VariationalTaus:
    a: float
    c: float
    d: NumpyArray1D


@dataclass
class VariationalParams:
    beta: VariationalBeta
    zs: VariationalZs
    taus: Optional[VariationalTaus] = None


class PriorType(Enum):
    NORMAL = 1
    NORMAL_GAMMA = 2
