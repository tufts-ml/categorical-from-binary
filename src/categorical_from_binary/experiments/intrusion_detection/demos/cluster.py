import faulthandler

from categorical_from_binary.datasets.cyber.data_frame import load_human_process_start_df
from categorical_from_binary.datasets.cyber.featurize import (
    construct_features,
    construct_labels,
)
from categorical_from_binary.datasets.cyber.util import (
    compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id,
    construct_minimal_process_id_by_process_id,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.pandas_helpers import keep_df_rows_by_column_values


faulthandler.enable()

### Load all human process start data
PATH_HUMAN_STARTS = (
    "/cluster/tufts/hugheslab/mwojno01/data/human_process_starts.csv"  # 1.5 GB
)
df_all = load_human_process_start_df(PATH_HUMAN_STARTS)

minimal_process_id_by_process_id = construct_minimal_process_id_by_process_id(df_all)
K = compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
    minimal_process_id_by_process_id
)
# `K` is the number of unique processes


### Get data for one user
USER_CBCMAIN_WITH_FEW_EVENTS = "U12279@DOM1"
USER_CBCMAIN_WITH_MANY_EVENTS = "U2863@DOM1"
# "U2863@DOM1"  has about 3000 observations, IIRC, and prior to sparsifying the design matrix,
# it segfaulted, at XtX construction.
user_domain = USER_CBCMAIN_WITH_MANY_EVENTS
df = keep_df_rows_by_column_values(df_all, col="user@domain", values=user_domain)

###
# Try to free memory
###
del df_all
import gc


gc.collect()


###
# Set configurables
###

K = 24742  # num processes
window_size = 5
temperature = 60  # one minute

###
# Featurize the dataset
###
labels = construct_labels(df, minimal_process_id_by_process_id, window_size)
features = construct_features(
    df,
    minimal_process_id_by_process_id,
    window_size,
    temperature,
    include_intercept=True,
)

####
# Run inference
####


# Prep training / test split
N = features.shape[0]
n_train_samples = int(0.8 * N)
covariates_train = features[:n_train_samples]
labels_train = labels[:n_train_samples]
covariates_test = features[n_train_samples:]
labels_test = labels[n_train_samples:]


print(f"About to run CAVI on dataset with {N:,} samples and {K:,} categories")

results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=20,
    # convergence_criterion_drop_in_mean_elbo=0.01,
)
print(f"\n\nPerformance over time: \n {results.performance_over_time}")
