"""
Plots illustrating how the CBC and CBM models differ in how the linear predictor
impacts category "potentials" (unnormalized probabilities).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns


cdf = scipy.stats.norm.cdf
inv_cdf = scipy.stats.norm.ppf


def compute_do_potential_from_linear_predictor(eta):
    """eta is the linear predictor, covariates dotted with regression weights"""
    return cdf(eta) / cdf(-eta)


def compute_sdo_potential_from_linear_predictor(eta):
    """eta is the linear predictor, covariates dotted with regression weights"""
    return cdf(eta)


def compute_do_potential_from_sdo_potential(y):
    """y = cdf(eta) is the sdo potential"""
    eta = inv_cdf(y)
    return compute_do_potential_from_linear_predictor(eta)


"""

"""

linear_predictors = np.linspace(-5, 10, 1000)
do_potentials = [
    compute_do_potential_from_linear_predictor(x) for x in linear_predictors
]
sdo_potentials = [
    compute_sdo_potential_from_linear_predictor(x) for x in linear_predictors
]

d = {
    "linear_predictors": pd.Series(list(linear_predictors) + list(linear_predictors)),
    "potentials": pd.Series(do_potentials + sdo_potentials),
    "links": pd.Series(["CBC"] * len(do_potentials) + ["CBM"] * len(sdo_potentials)),
}
df = pd.DataFrame(d)


#### plots
sns.lineplot(data=df, x="linear_predictors", y="potentials", hue="links", legend=True)
plt.xlim(-3, 10)
plt.ylim(0, 10)
plt.show()

sdo_potentials = np.linspace(1 / 1000, 1.0 - 1 / 1000, 1000)
do_potentials = [compute_do_potential_from_sdo_potential(x) for x in sdo_potentials]

plt.plot(sdo_potentials, do_potentials)
plt.xlabel("sdo potential")
plt.ylabel("do potential")
plt.show()
