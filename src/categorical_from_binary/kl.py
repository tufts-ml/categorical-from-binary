import numpy as np
from scipy.special import digamma, multigammaln

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def compute_kl_mvn(
    m0: NumpyArray1D,
    S0: NumpyArray2D,
    m1: NumpyArray1D,
    S1: NumpyArray2D,
) -> float:
    """
    Computes the Kullback-Liebler divergence from Gaussian p1 (with mean m1, var S1)
    to Gaussian p0 (with mean m0, var S0).

    From wikipedia
    KL(p0||p1)
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )

    References:
        https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv/55688087
        https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    """
    # TODO: This function must exist somewhere in a python library.  It would be better to import
    # and use that; no point in reinventing the wheel.  Said function would already be unit tested,
    # would be more flexible to input types, etc.

    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))
    quad_term = diff.T @ iS1 @ diff
    return 0.5 * (tr_term + det_term + quad_term - N)


def compute_kl_inverse_wishart(
    v1: float,
    S1: NumpyArray2D,
    v2: float,
    S2: NumpyArray2D,
) -> float:
    """
    Computes the Kullback-Liebler divergence from Inverse Wishart p2 (with dof v2, residual sum of squares S2)
    to Inverse Wishart p1 (with dof v1, residual sum of squares S1).

    References:
        Maya Gupta and Santosh Srivastava. Parametric bayesian estimation of differential entropy and relative
            entropy.  Entropy, 12(4):818--843, 2010.
        Wojnowicz, Michael.  Exponential families.  Available upon request.
    """
    # TODO: Write unit test or find external library to call.
    # Example unit test:  For fixed degrees of freedom, as means diverge, so should KL divergence.
    # For fixed means, KL divergence should increase as d.o.f. (i.e certainty) v1 and v2 spread farther and
    # father apart

    # precomputations
    N = S1.shape[0]
    iS1 = np.linalg.inv(S1)
    iS1xS2 = iS1 @ S2

    # kl is made of five terms
    tr_term = 0.5 * v1 * np.trace(iS1xS2)
    det_term = 0.5 * v2 * np.log(np.linalg.det(iS1xS2))
    gamma_term = multigammaln(0.5 * v2, N) - multigammaln(0.5 * v1, N)
    digamma_term = (
        0.5 * (v2 - v1) * np.sum([digamma(0.5 * (v1 - N + i)) for i in range(1, N + 1)])
    )
    simple_term = 0.5 * v1 * N

    return gamma_term + tr_term - simple_term - det_term - digamma_term


def compute_expected_log_det_of_inverse_wishart_rv(
    v: float,
    S: NumpyArray2D,
) -> float:
    """
    Arguments:
        v : degrees of freedom
        S : Residual sum of squares
    """
    # TODO: Write unit test

    N = S.shape[0]
    return (
        -N * np.log(2)
        + np.log(np.linalg.det(S))
        - np.sum([digamma(0.5 * (v - N + i)) for i in range(1, N + 1)])
    )


def compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument(
    mu_0: NumpyArray1D,
    Sigma_0: NumpyArray2D,
    m: NumpyArray1D,
    V: NumpyArray2D,
    nu: float,
    S: NumpyArray2D,
) -> float:
    """
    Gives the expected KL divergence between two multivariate Gaussians,
        E_q(mu_1, Sigma_1)  KL (  p_0(X | mu_0, Sigma_0)  || p_1(X | mu_1, Sigma_1))
    in the specific case where the expectation is taken with respect to independent Gaussian
    and Inverse Wishart distributions on the parameters of the second argument of the KL divergence.

    We notate parameters by
        q(mu_1, Sigma_1) =  q(mu_1 | m, V) q(Sigma_1 | nu,  S)

    Arguments:
        mu_0 : NumpyArray1D,
            The mean parameter of the Gaussian of the first argument to the KL divergence
        Sigma_0 : NumpyArray2D,
            The covariance parameter of the Gaussian of the first argument to the KL divergence
        m : NumpyArray1D,
            The mean parameter of the Gaussian distribution on mu_1,  which is the mean
            of the Gaussian of the second argument to the KL divergence
        V : NumpyArray2D,
            The covariance parameter of the Gaussian distribution on mu_1,  which is the mean
            of the Gaussian of the second argument to the KL divergence
        nu : float,
            The degrees of freedom parameter of the Inverse Wishart distribution on Sigma_1,
            which is the covariance of the Gaussian of the second argument to the KL divergence
        S : NumpyArray2D,
            The residual sum of squares parameter of the Inverse Wishart distribution on Sigma_1,
            which is the covariance of the Gaussian of the second argument to the KL divergence
    """
    # TODO: Write checks that dimensionalities match

    N = S.shape[0]

    return 0.5 * (
        compute_expected_log_det_of_inverse_wishart_rv(nu, S)
        - np.log(np.linalg.det(Sigma_0))
        - N
        + np.trace(
            nu
            * np.linalg.inv(S)
            @ (mu_0 @ mu_0.T - mu_0 @ m.T - m @ mu_0.T + V + m @ m.T + Sigma_0)
        )
    )
