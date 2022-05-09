"""
Round robin scoring takes as inputs the models (just the posterior means) from training, 
and uses them to score data from all the users for which there were models.
So if there are U users, then this procedure produces U^2 scores.

We also plot the round robin results in heatmap (self vs other) form.
"""

import glob
import os

import numpy as np
import pandas as pd
import scipy

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.datasets.cyber.data_frame import load_human_process_start_df
from categorical_from_binary.datasets.cyber.featurize import (
    construct_features,
    construct_labels,
)
from categorical_from_binary.datasets.cyber.util import (
    compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id,
    compute_sorted_sample_sizes_for_users,
    construct_minimal_process_id_by_process_id,
)
from categorical_from_binary.metrics import compute_metrics
from categorical_from_binary.pandas_helpers import keep_df_rows_by_column_values


### configs
training_results_dir = "/cluster/tufts/hugheslab/mwojno01/data/results/"
scoring_results_df_path = "/cluster/tufts/hugheslab/mwojno01/data/results/mean_log_likes_from_fifty_user_round_robin.csv"

PATH_TO_HUMAN_STARTS = (
    "/cluster/tufts/hugheslab/mwojno01/data/human_process_starts.csv"  # 1.5 GB
)

filepaths = glob.glob(training_results_dir + "*@DOM1*")
user_domains = list(set([fp.split("/")[-1].split("_")[0] for fp in filepaths]))
# TODO: Need to be careful in running on all users with paths!
# E.g. U8048@DOM1 has no data for some reason. it's probably because that user has an abnormal
# domain for this set. i'm not sure how to enforce that the directory won't get populated with "junk"


window_size = 5  # TODO: this should be obtained from filename and/or metadata
temperature = 60  # TODO: this should be obtained from filename and/or metadata
start_idx_for_users_in_subset = (
    300  # TODO: this should be obtained from filename and/or metadata
)
number_of_users_in_subset = (
    50  # TODO: this should be obtained from filename and/or metadata
)

###
# Load data subset used in training to determine the number of categories and the mapping
# from the original data representation to our {1,...,K}.
###
# TODO: This info should just get saved at training time, instead of having to reconstruct it
# from now-lost run parameters.

df_all = load_human_process_start_df(PATH_TO_HUMAN_STARTS)

# get a subset of all user domains, from which we determine the number of categories to use for processing.
sorted_sample_sizes_for_users = compute_sorted_sample_sizes_for_users(df_all)
user_domains_in_subset = sorted_sample_sizes_for_users.index[
    start_idx_for_users_in_subset : start_idx_for_users_in_subset
    + number_of_users_in_subset
]
df_subset = keep_df_rows_by_column_values(
    df_all, col="user@domain", values=user_domains_in_subset
)
minimal_process_id_by_process_id = construct_minimal_process_id_by_process_id(df_subset)
K = compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
    minimal_process_id_by_process_id
)
print("The total number of distinct processes started in this subset of users is {K}.")

N_users = len(user_domains)
mean_log_likes = np.zeros((N_users, N_users))

###
# Make (model user x data user) log likelihood matrix
####

# TEMPORARY_RESTRICTION_ON_NUM_USERS =
### load  model from model user
for model_user_idx in range(len(user_domains)):
    model_user_domain = user_domains[model_user_idx]
    beta_mean_path = os.path.join(
        training_results_dir, f"{model_user_domain}_beta_mean.npz"
    )
    beta_mean = scipy.sparse.load_npz(beta_mean_path)

    ### load data from data user
    for data_user_idx in range(len(user_domains)):
        print(
            f"Now scoring user {data_user_idx+1}/{N_users} with user model {model_user_idx+1}/{N_users}"
        )
        user_domain = user_domains[data_user_idx]
        df = keep_df_rows_by_column_values(
            df_all, col="user@domain", values=user_domain
        )

        ###
        # Featurize the other dataset
        ###
        labels = construct_labels(df, minimal_process_id_by_process_id, window_size)
        features = construct_features(
            df,
            minimal_process_id_by_process_id,
            window_size,
            temperature,
            include_intercept=True,
        )

        N = features.shape[0]
        n_train_samples = int(0.8 * N)

        try:
            covariates_train = features[:n_train_samples]
            labels_train = labels[:n_train_samples]
            covariates_test = features[n_train_samples:]
            labels_test = labels[n_train_samples:]

            # TODO: Add BMA!
            probs = construct_category_probs(
                features,
                beta_mean,
                Link.CBC_PROBIT,
            )
            metrics = compute_metrics(probs, labels)
            mean_log_like = metrics.mean_log_like
            print(f"\t The mean log like was {mean_log_like}")
            mean_log_likes[model_user_idx, data_user_idx] = mean_log_like
        except:  # some users don't have data for some reason
            print(
                f"Could not process model_user_idx, data_user_idx = {model_user_idx}, {data_user_idx}"
            )
            mean_log_likes[model_user_idx, data_user_idx] = np.nan

df_mean_log_likes = pd.DataFrame(mean_log_likes, user_domains, user_domains)
df_mean_log_likes.to_csv(scoring_results_df_path)
