"""
Goal: Make plot that shows one curve for each category prob
"""
from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.experiments.marksmanship.analysis.covariate_modulation import (
    make_covariate_modulation_dict_for_feature_using_CAVI_model,
    make_covariate_modulation_dict_for_feature_using_SKLEARN_model,
    make_x_to_z,
    make_z_to_x,
)
from categorical_from_binary.types import NumpyArray2D


sns.set_theme()


def _plot_covariate_modulation_dict(
    covariate_dict,
    feature_name,
    covariate_modulation_dict,
    y_bottom: Optional[float] = None,
    y_top: Optional[float] = None,
):
    df = covariate_modulation_dict
    is_binary = covariate_dict[feature_name].type.name == "BINARY"

    ax = sns.lineplot(data=df, x="covariate", y="prob", hue="response")
    ax.set_xlabel(f"raw value of {feature_name}")
    ax.set_ylabel("probability of response")
    ax.set_ylim(y_bottom, y_top)
    if is_binary:
        # enforce only two ticks on secondary axis
        ax.set_xticks([0, 1])
    ### Plot secondary axis (with z-score)
    #   Reference: https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/secondary_axis.html
    if not is_binary:
        x_to_z = make_x_to_z(covariate_dict, feature_name)
        z_to_x = make_z_to_x(covariate_dict, feature_name)
        secax = ax.secondary_xaxis("top", functions=(x_to_z, z_to_x))
        secax.set_xlabel(f"z-score of {feature_name}")

    ### Put the legend out of the figure
    # Reference: https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
    ax.legend(title="response", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    return ax


def plot_response_probabilities_by_covariate_value_using_CAVI_model(
    covariate_dict,
    label_dict,
    beta_mean: NumpyArray2D,
    feature_name: str,
    category_probability_link: Link,
    y_bottom: Optional[float] = None,
    y_top: Optional[float] = None,
):

    covariate_modulation_dict = (
        make_covariate_modulation_dict_for_feature_using_CAVI_model(
            covariate_dict,
            label_dict,
            beta_mean,
            feature_name,
            category_probability_link,
        )
    )
    return _plot_covariate_modulation_dict(
        covariate_dict,
        feature_name,
        covariate_modulation_dict,
        y_bottom,
        y_top,
    )


def plot_response_probabilities_by_covariate_value_using_SKLEARN_model(
    covariate_dict,
    label_dict,
    sklearn_model: LogisticRegression,
    feature_name: str,
    y_bottom: Optional[float] = None,
    y_top: Optional[float] = None,
):

    covariate_modulation_dict = (
        make_covariate_modulation_dict_for_feature_using_SKLEARN_model(
            covariate_dict,
            label_dict,
            sklearn_model,
            feature_name,
        )
    )
    return _plot_covariate_modulation_dict(
        covariate_dict,
        feature_name,
        covariate_modulation_dict,
        y_bottom,
        y_top,
    )
