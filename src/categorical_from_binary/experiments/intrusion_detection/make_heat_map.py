# WORK LOCALLY, not on cluster!!
# on cluster: results at "/cluster/tufts/hugheslab/mwojno01/data/results/intrusion/"

import os

import matplotlib
import numpy as np
import pandas as pd

# from matplotlib import colors
from matplotlib import pyplot as plt


matplotlib.rc_file_defaults()


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
# Plot
###
# configs

# compute metric

methods = ["advi", "cavi"]
method_labels = ["ADVI (200 min)", "IB-CAVI (20 min)"]
N_users = len(list(active_directory_mean_log_likes_by_method.values())[0])

metrics = np.zeros((N_users, 2))
for (m, method) in enumerate(methods):
    values = active_directory_mean_log_likes_by_method[method]
    for u in range(N_users):
        values_for_model = values[u]
        self_score = values_for_model[u]
        values_for_model_without_self = np.delete(values_for_model, u)
        metrics[u, m] = self_score - np.mean(values_for_model_without_self)

indices_sorted = np.argsort(metrics[:, 0])
metrics_sorted = metrics[np.ix_(indices_sorted, [0, 1])]


cmap = "RdBu_r"  #'RdYlGn' # 'RdBu_r'  # matplotlib.cm.viridis
fig, ax = plt.subplots(1, 1, figsize=(8, 2), squeeze=True, constrained_layout=True)
# my_norm=colors.TwoSlopeNorm(vcenter=0)
# im = ax.imshow(metrics_sorted.T, cmap=cmap, norm=my_norm)
im = ax.imshow(metrics_sorted.T, cmap=cmap)

bar = plt.colorbar(im)
bar.set_label("Intrusion Detection Score")
plt.xlabel("Model owner")
plt.ylabel("Method")
ax.set_yticks([0, 1])
ax.set_yticklabels(method_labels)
plt.show()

###
# Statistics
###

import scipy.stats


scipy.stats.ttest_rel(metrics_sorted[:, 1], metrics_sorted[:, 0])

# TODO: Make table
