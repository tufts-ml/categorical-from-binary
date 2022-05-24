# WORK LOCALLY, not on cluster!!
# on cluster: results at "/cluster/tufts/hugheslab/mwojno01/data/results/intrusion/"

import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


matplotlib.rc_file_defaults()
import seaborn as sns


sns.set(style="whitegrid")


def model_user_domain_from_basename(basename):
    return basename.split("model_user=")[1].split("_")[0]


def get_model_user_domains(local_results_dir):
    return [
        model_user_domain_from_basename(basename)
        for basename in os.listdir(local_results_dir)
    ]


def get_grand_mean_log_likes_with_user_domain_labels(
    local_results_dir: str, cavi_or_advi_string: str
):
    """
    Returns:
        grand_mean_log_likes: np.array, (i,j)th entry gives mean log like when the i-th user model
            scores holdout data from the j-th user.
        user_domains
    """
    user_domains = sorted(set(get_model_user_domains(local_results_dir)))
    N_users = len(user_domains)
    grand_mean_log_likes = np.zeros((N_users, N_users))  # model user X data user
    for basename in os.listdir(local_results_dir):
        if cavi_or_advi_string in basename:
            full_path_for_one_user_model = os.path.join(local_results_dir, basename)
            df_mean_log_likes_for_one_user_model = pd.read_csv(
                full_path_for_one_user_model, index_col=0
            )
            model_user_domain = model_user_domain_from_basename(basename)
            model_user_idx = user_domains.index(model_user_domain)
            for (
                data_user_domain,
                mean_log_like,
            ) in df_mean_log_likes_for_one_user_model.squeeze().iteritems():
                data_user_idx = user_domains.index(data_user_domain)
                grand_mean_log_likes[model_user_idx, data_user_idx] = mean_log_like
    return grand_mean_log_likes, user_domains


def restrict_grand_mean_log_likes_with_user_domain_labels_to_active_directory_domains(
    grand_mean_log_likes, user_domains
):
    """
    Active directory domain account are the user_domains which end with @DOM1.
    """
    active_directory_indices = [
        i for (i, ud) in enumerate(user_domains) if "@DOM1" in ud
    ]
    active_directory_user_domains = [user_domains[i] for i in active_directory_indices]
    active_directory_mean_log_likes = grand_mean_log_likes[
        np.ix_(active_directory_indices, active_directory_indices)
    ]
    # np.ix_ does subsetting via a cartesian product.
    return active_directory_mean_log_likes, active_directory_user_domains


active_directory_mean_log_likes_by_method = {}
for cavi_or_advi_string in ["cavi", "advi"]:
    local_results_dir = "/Users/mwojno01/Repos/categorical-from-binary/data/results/intrusion/intrusion_cavi_20_mins_vs_advi_200_mins/"
    (
        grand_mean_log_likes,
        user_domains,
    ) = get_grand_mean_log_likes_with_user_domain_labels(
        local_results_dir, cavi_or_advi_string
    )
    (
        active_directory_mean_log_likes_for_this_method,
        active_directory_user_domains,
    ) = restrict_grand_mean_log_likes_with_user_domain_labels_to_active_directory_domains(
        grand_mean_log_likes, user_domains
    )
    active_directory_mean_log_likes_by_method[
        cavi_or_advi_string
    ] = active_directory_mean_log_likes_for_this_method


###
# Intrusion detection metric
###

methods = ["advi", "cavi"]
method_labels = ["Softmax+ADVI [200 min]", "CB-Probit+IB-CAVI [20 min]"]
# method_labels = ["ADVI (3 hr 20 min)", "IB-CAVI (20 min)"]
# try to match color and alpha of perf over time plot
color_for_advi = sns.color_palette(palette="BuPu", n_colors=4)[3]
color_for_cavi = sns.color_palette(palette="YlOrRd", n_colors=2)[-1]
colors = [color_for_advi, color_for_cavi]

alpha_for_advi = 0.5
alpha_for_cavi = 1.0
alphas = [alpha_for_advi, alpha_for_cavi]

N_users = len(list(active_directory_mean_log_likes_by_method.values())[0])
metrics = np.zeros((N_users, 2))
for (m, method) in enumerate(methods):
    values = active_directory_mean_log_likes_by_method[method]
    for u in range(N_users):
        values_for_model = values[u]
        self_score = values_for_model[u]
        values_for_model_without_self = np.delete(values_for_model, u)
        metrics[u, m] = self_score - np.mean(values_for_model_without_self)

indices_sorted = np.argsort(metrics[:, 1])
metrics_sorted = metrics[np.ix_(indices_sorted, [0, 1])]


###
# PLot
###
fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True, constrained_layout=True)
# my_norm=colors.TwoSlopeNorm(vcenter=0)
# im = ax.imshow(metrics_sorted.T, cmap=cmap, norm=my_norm)
for i in [1, 0]:
    plt.plot(
        np.arange(1, N_users + 1),
        metrics_sorted[:, i],
        label=method_labels[i],
        linewidth=4,
        color=colors[i],
        alpha=alphas[i],
    )

# add legend
legend = plt.legend(
    shadow=True,
    fancybox=True,
    # bbox_to_anchor=(1, 0.5, 0.3, 0.2),
    loc="upper left",
    borderaxespad=0,
    fontsize=24,
)


plt.xlabel("User model", fontsize=32)
plt.ylabel("Intrusion Detection Score", fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.set_xticks([1, 8, 16, 24, 32])
ax.set_xticklabels([1, 8, 16, 24, 32])
plt.show()

###
# Statistics
###

import scipy.stats


scipy.stats.ttest_rel(metrics_sorted[:, 1], metrics_sorted[:, 0])

# TODO: Make table
