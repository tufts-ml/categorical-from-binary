import numpy as np

from categorical_from_binary.types import NumpyArray2D


def compute_probs_for_baserate_model(labels_train: NumpyArray2D, n_samples: int):
    """
    We create a baserate model just looks at the frequency of each category in the training set and then
    predicts categories in that order for every sample, regardless of covariate.
    """
    probs_for_each_sample = np.mean(labels_train, 0)
    return np.repeat(probs_for_each_sample[:, np.newaxis], n_samples, axis=1).T
