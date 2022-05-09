import numpy as np


def compute_polya_gamma_expectation(
    b: float, c: float, allow_b_to_be_zero: bool = False
) -> float:
    """
    The code implements the top right equation on pp. 1341 of
    Polson, Scott, Windle (2013), JASA,  with the exception that the implementation
    bounds c away from 0 with an epsilon-ball.  This is due to the fact that the
    paper's provided expression for the expectation is is undefined at b==0, even
    though the expectation is actually defined.  (I have not proven that it is, but I have
    this on supposition based on Monte Carlo approximations combined with a first foray into
    proving it.)

    Arguments:
        b:  first parameter of the Polya-Gamma distribution.  Must be >0.
        c:  second parameter of the Polya-Gamma distribution.  Must be real-valued.
        allow_b_to_be_zero:  False by default, due to the support of the Polya-Gamma distribution.
            If True, then we always return an expectation of 0 when b equals 0, which has computational
            advantages.
    Reference:
        Polson, Scott, Windle,  JASA,  2013.
    """
    if b < 0:
        raise ValueError(f"b={b}, but it cannot be less than 0")
    if b == 0 and not allow_b_to_be_zero:
        raise ValueError(
            f"b={b}, but it cannot equal 0, unless the allow_b_to_be_zero flag is set to True"
        )
    if b == 0 and allow_b_to_be_zero:
        return 0

    # If c is too close to 0, we replace it with epsilon, because the paper's formula
    # is undefined at c==0, even though the expectation has a known value.
    EPSILON = 1e-10
    if abs(c) < EPSILON:
        c = EPSILON

    return b / (2 * c) * (np.exp(c) - 1) / (1 + np.exp(c))
