import numpy as np


###
# Categorical
###


def _soften_probs(probs: np.ndarray, epsilon=1e-6) -> np.ndarray:
    """
    Motivations:
        * sample_from_categorical function returns error if it is too close to a vertex
            (e.g., [1,0,0], in the case of 3 categories)
        * sample_from_categorical can error out if the sum is slightly different than 1.0,
            and in fact dividing the unnormalized probabilities by the normalizing constant
            does not fix the problem (the sum can still be slightly different than 1.0)
    """
    probs[probs < epsilon] = 0.0
    return probs / sum(probs)


def sample_from_categorical(probs: np.ndarray, one_indexed: bool = False) -> float:
    """
    Returns a (numerically-labeled) category sampled from a categorical distribution.
    """
    probs = _soften_probs(probs)
    category = np.argmax(np.random.multinomial(n=1, pvals=probs) > 0)
    # TODO:  sample_from_categorical seems to fails when some probabilities get too close
    # to 0 or 1, and the issue traces back to the call to np.random.multinomial:
    """
        70     probs = _soften_probs(probs)
    ---> 71     category = np.argmax(np.random.multinomial(n=1, pvals=probs) > 0)
        72     if one_indexed:
        73         category += 1

    mtrand.pyx in numpy.random.mtrand.RandomState.multinomial()

    _common.pyx in numpy.random._common.check_array_constraint()

    ValueError: pvals < 0, pvals > 1 or pvals contains NaNs
    """
    if one_indexed:
        category += 1
    return category
