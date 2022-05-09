import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from pypolyagamma import PyPolyaGamma

from categorical_from_binary.polya_gamma.polya_gamma import compute_polya_gamma_expectation


@given(
    b=st.floats(min_value=1, max_value=1),
    c=st.floats(min_value=-100, max_value=100),
)

# The `b` parameter was fixed at 1.0 because it is always 1.0 for logistic regression
# (It can be larger than one when doing binomial regression more generally).
# I didn't really have a good reason for setting the `c` limits,  other than having them
# not get too large based on the fact that c = x_i^T beta.  We might want to ask the user (or provide a routine)
# for normalizing continuous covariates before feeding into the algorithm,  as very large values may cause issues.


def test_compute_polya_gamma_expectation(b, c):

    # Test that our computed polya gamma expectation for a PG(b,c) distribution
    # is close to the empirical mean of a bunch of samples (obtained from the pypolyagamma library)

    pg = PyPolyaGamma()
    empirical_mean = np.mean([pg.pgdraw(b, c) for i in range(10000)])
    computed_mean = compute_polya_gamma_expectation(b, c)
    # print(
    #     f"For b={b}, c={c}, the Monte Carlo mean was {empirical_mean}, and my function's value was {computed_mean}"
    # )
    assert np.isclose(empirical_mean, computed_mean, atol=0.01, rtol=0.05)
