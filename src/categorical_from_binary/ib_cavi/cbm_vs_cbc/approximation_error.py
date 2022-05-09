import numpy as np

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def compute_approximation_error_in_category_probs_using_l1_distance(
    cat_probs_1: NumpyArray2D,
    cat_probs_2: NumpyArray2D,
) -> NumpyArray1D:
    """
    cat probs have shape (N,K), where N is the number of samples and K is the number of categories.
    """
    return np.sum(np.abs(cat_probs_1 - cat_probs_2), 1)


def compute_mean_approximation_error_in_category_probs_using_l1_distance(
    cat_probs_1: NumpyArray2D,
    cat_probs_2: NumpyArray2D,
) -> NumpyArray1D:
    """
    cat probs have shape (N,K), where N is the number of samples and K is the number of categories.
    """
    return np.mean(np.abs(cat_probs_1 - cat_probs_2), 1)


def compute_signed_error_in_largest_category_probs(
    cat_probs_estimated: NumpyArray2D,
    cat_probs_ground_truth: NumpyArray2D,
) -> NumpyArray1D:
    predicted_categories = np.argmax(cat_probs_estimated, 1)
    # TODO: there should be a simpler way to write the below.  I basically want
    # cat_probs_estimated[:,predicted_categories]-cat_probs_ground_truth[:,predicted_categories]
    return np.array(
        [cat_probs_estimated[i, k] for (i, k) in enumerate(predicted_categories)]
    ) - np.array(
        [cat_probs_ground_truth[i, k] for (i, k) in enumerate(predicted_categories)]
    )


def compute_kl_divergence_from_estimated_to_true_category_probs(
    cat_probs_estimated: NumpyArray2D,
    cat_probs_true: NumpyArray2D,
    epsilon: float = 1e-20,
) -> NumpyArray1D:
    """
    Arguments:
        cat_probs_estimated: has shape (N,K), where N is the number of samples and K is the number of categories.
        cat_probs_true: has shape (N,K), where N is the number of samples and K is the number of categories.
        epsilon: minimum value for a probability component; must be bounded away from 0, because the KL
            divergence requires computing log(p_ik), which is NaN if p_ik=0.
    Returns:
        array of shape (N,)
    """
    # lower bound probability vector components to be slightly away from zero, because KL divergence requires
    # computing log(p_ik)
    for cat_probs in [cat_probs_estimated, cat_probs_true]:
        cat_probs += epsilon
        cat_probs /= np.sum(cat_probs, 1)[:, np.newaxis]

    return np.sum(
        cat_probs_true * (np.log(cat_probs_true) - np.log(cat_probs_estimated)), 1
    )
