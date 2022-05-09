from dataclasses import dataclass
from typing import Optional


@dataclass
class MetaData:
    n_categories: int
    mean_log_like_random_guessing: float
    accuracy_random_guessing: float
    mean_log_like_data_generating_process: Optional[float] = None
    accuracy_data_generating_process: Optional[float] = None
