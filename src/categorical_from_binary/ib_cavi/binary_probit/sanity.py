import numpy as np
from scipy.stats import norm as norm
from tabulate import tabulate


def compute_class_probabilities(covariates, beta):
    linear_predictors = covariates @ beta
    return norm.cdf(linear_predictors)


def print_table_comparing_class_probs_to_labels(class_probs, labels):
    titles = [
        "training data label",
        "modeled class probs",
    ]
    results = np.transpose([class_probs, labels])
    table = tabulate(results, titles, tablefmt="fancy_grid")
    print(table)
