import numpy as np


def logdiffexp(a, b):
    """log(exp(a) - exp(b)).  Borrowed from pymc3 library."""
    return a + log1mexp(b - a)


def log1mexp(x):
    """Return log(1 - exp(x)).
    This function is numerically more stable than the naive approach.
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    Borrowed from pymc3 library.
    """
    x = np.asarray(x, dtype="float")
    if not (x <= 0).all():
        raise ValueError("All components of x must be negative.")
    out = np.empty_like(x)
    mask = x < -0.6931471805599453  # log(1/2)
    out[mask] = np.log1p(-np.exp(x[mask]))
    mask = ~mask
    out[mask] = np.log(-np.expm1(x[mask]))
    return out
