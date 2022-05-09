from enum import Enum

from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.datasets.generic.detergent.load import (
    construct_detergent_data_split,
    load_detergent_data,
)
from categorical_from_binary.datasets.generic.frogs.load import (
    construct_frog_data_split,
    load_frog_data,
)
from categorical_from_binary.datasets.generic.glass.load import (
    construct_glass_identification_data_split,
    load_glass_identification_data,
)


class Dataset(int, Enum):
    DETERGENT = 1
    FROGS = 2
    GLASS = 3


DATASET_TO_LOADING_FUNCTION = {
    Dataset.DETERGENT: load_detergent_data,
    Dataset.FROGS: load_frog_data,
    Dataset.GLASS: load_glass_identification_data,
}

DATASET_TO_DATA_SPLITTER_FUNCTION = {
    Dataset.DETERGENT: construct_detergent_data_split,
    Dataset.FROGS: construct_frog_data_split,
    Dataset.GLASS: construct_glass_identification_data_split,
}


def construct_data_split(
    dataset: Dataset,
    pct_training: int,
    standardize_design_matrix: bool,
    random_seed: int,
) -> SplitDataset:

    load_dataset = DATASET_TO_LOADING_FUNCTION[dataset]
    split_dataset = DATASET_TO_DATA_SPLITTER_FUNCTION[dataset]

    data = load_dataset()

    dataset_split = split_dataset(
        data,
        pct_training=pct_training,
        standardize_design_matrix=standardize_design_matrix,
        random_seed=random_seed,
    )

    return dataset_split
