from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame


def create_hot_pairplot(df: DataFrame, n_bins=20) -> sns.axisgrid.PairGrid:
    """
    Create a pairwise plot where the diagonals show histograms and the off diagonals show
    heatmaps.

    The more direct way of creating pair plots, via `sns.pairplot(df)`, has scatterplots on the
    off-diagonals, which is hard to interpret for large datasets, due to points overlapping.

    Returns:
        Subplot grid for plotting pairwise relationships in a dataset.
        Can be shown by calling plt.show() after runnign the function

    Reference:
        https://stackoverflow.com/questions/43924280/pair-plot-with-heat-maps-possibly-logarithmic?rq=1

    """
    g = sns.PairGrid(df)
    g.map_diag(plt.hist, bins=n_bins)

    def pairgrid_heatmap(x, y, **kws):
        cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
        plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)

    g.map_offdiag(pairgrid_heatmap, bins=n_bins)
    return g


def make_linear_regression_plot(
    df: DataFrame,
    x_col_name: str,
    y_col_name: str,
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    reg_line_label: Optional[str] = None,
    add_y_equals_x_line=True,
    add_y_equals_0_line=True,
    lowess=False,
    ax=None,
    plot_points_as_density_not_scatter=True,
):
    """
    Arguments:
        plot_points_as_density_not_scatter:  Defaults to True because a density plot is much
            easier to grok than a scatter plot when there are lots of points - can more easily tell
            how unlikely it is to get an outlier value.
    """
    if ax is None:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 6))

    if plot_points_as_density_not_scatter:
        cmap = sns.light_palette("seagreen", as_cmap=True)
        counts, xedges, yedges, im = ax.hist2d(
            df[x_col_name], df[y_col_name], cmap=cmap, cmin=1
        )
        plt.colorbar(im, ax=ax)

    else:
        ax.scatter(df[x_col_name], df[y_col_name], color="red", alpha=0.5)

    if add_y_equals_0_line:
        ax.hlines(
            y=0,
            xmin=min(df[x_col_name]),
            xmax=max(df[x_col_name]),
            colors="k",
            linestyles="--",
        )

    sns.regplot(
        x=x_col_name,
        y=y_col_name,
        data=df,
        fit_reg=True,
        ci=None,
        label=reg_line_label,
        scatter=False,
        color="black",
        lowess=lowess,
        ax=ax,
    )
    if add_y_equals_x_line:
        sns.regplot(
            x=x_col_name,
            y=x_col_name,
            data=df,
            fit_reg=True,
            ci=None,
            label="y=x",
            scatter=False,
            color="black",
            ax=ax,
        )
    ax.yaxis.set_tick_params(labelleft=True)
    ax.set(ylabel=y_axis_label, xlabel=x_axis_label)
    ax.set_title(title)
    return ax
