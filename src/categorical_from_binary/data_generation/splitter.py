from typing import NamedTuple, Tuple

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    MulticlassRegressionDataset,
)
from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    HierarchicalMulticlassRegressionDataset,
)
from categorical_from_binary.types import NumpyArray2D


class SplitDataset(NamedTuple):
    """
    covariates_train: numpy array with shape (N,M), where N is the number of samples and
        M is the number of covariates.   If None, the model will automatically populate it
        with a (N,1) vector of all 1's in order to run an intercepts-only model
    labels_train: an np.array with shape (N,K), where N is a number of samples and K is the
        number of categories.  position (n,k) = 1 if the n-th sample used the k-th category, and is 0 otherwise.
        Thus, summing across columns produces a (N,)-shaped vector of all 1's.
    covariates_test: numpy array with shape (N,M), where N is the number of samples and
        M is the number of covariates.   If None, the model will automatically populate it
        with a (N,1) vector of all 1's in order to run an intercepts-only model
    labels_test: an np.array with shape (N,K), where N is a number of samples and K is the
        number of categories.  position (n,k) = 1 if the n-th sample used the k-th category, and is 0 otherwise.
        Thus, summing across columns produces a (N,)-shaped vector of all 1's.
    """

    covariates_train: NumpyArray2D
    labels_train: NumpyArray2D
    covariates_test: NumpyArray2D
    labels_test: NumpyArray2D


def split_multiclass_regression_dataset(
    data: MulticlassRegressionDataset,
    n_train_samples: int,
) -> Tuple[MulticlassRegressionDataset, MulticlassRegressionDataset]:
    """
    Split a `MulticlassRegressionDataset` into a training dataset and a test dataset
    via a simple routine:  We take the first `n_train_samples` from each dataset
    """
    data_train = MulticlassRegressionDataset(
        data.features[:n_train_samples],
        data.labels[:n_train_samples],
        data.beta,
        data.link,
        data.seed,
    )
    data_test = MulticlassRegressionDataset(
        data.features[n_train_samples:],
        data.labels[n_train_samples:],
        data.beta,
        data.link,
        data.seed,
    )
    return data_train, data_test


def split_hierarchical_multiclass_regression_dataset(
    hd: HierarchicalMulticlassRegressionDataset, n_train_samples: int
) -> Tuple[
    HierarchicalMulticlassRegressionDataset, HierarchicalMulticlassRegressionDataset
]:
    """
    Split a `HierarchicalMulticlassRegressionDataset` into a training dataset and a test dataset
    via a simple routine:  We take the first `n_train_samples` from each dataset
    """
    datasets_split = [
        split_multiclass_regression_dataset(dataset, n_train_samples)
        for dataset in hd.datasets
    ]

    hierarchical_data_train = HierarchicalMulticlassRegressionDataset(
        [dataset_split[0] for dataset_split in datasets_split],
        hd.beta_expected,
        hd.beta_cov,
        hd.link,
        hd.seed,
        hd.is_autoregressive,
    )
    hierarchical_data_test = HierarchicalMulticlassRegressionDataset(
        [dataset_split[1] for dataset_split in datasets_split],
        hd.beta_expected,
        hd.beta_cov,
        hd.link,
        hd.seed,
        hd.is_autoregressive,
    )
    return hierarchical_data_train, hierarchical_data_test
