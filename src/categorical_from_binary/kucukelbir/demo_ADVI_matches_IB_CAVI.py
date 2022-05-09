from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.kucukelbir.inference import (
    Link2,
    Metadata,
    do_advi_inference_via_kucukelbir_algo,
)


###
# Construct dataset
###
n_categories = 3
n_features = 1
n_samples = 5000
include_intercept = True
link = Link.MULTI_LOGIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=2.0
)
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link,
    seed=None,
    include_intercept=include_intercept,
    beta_category_strategy=beta_category_strategy,
)


# Prep training / test split
n_train_samples = int(0.8 * n_samples)
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]


###
# ADVI inference
###

# configs
random_seed = 0
n_advi_iterations = 100
lr = 0.1
link2 = Link2.SOFTMAX

# do inference
metadata = Metadata(n_samples, n_features, n_categories, include_intercept)
(
    beta_mean_ADVI,
    beta_stds_ADVI,
    performance_ADVI,
) = do_advi_inference_via_kucukelbir_algo(
    labels_train,
    covariates_train,
    metadata,
    link2,
    n_advi_iterations,
    lr,
    random_seed,
    labels_test=labels_test,
    covariates_test=covariates_test,
)


# ###
# # Evaluate ADVI
# ###

# data_test = (X_test, y_test)
# log_like_holdout = compute_log_like(mus, data_test, full_train_size, metadata)
# mean_log_like_holdout = log_like_holdout / len(y_test)
# print(f"\nMean holdout log like: {mean_log_like_holdout:.02f}")

# # check accuracy
# beta_matrix_ADVI = beta_matrix_from_vector(
#     mus, metadata.M, metadata.K, include_intercept
# )
# cat_probs_test_ADVI = compute_CBC_Probit_predictions(X_test, beta_matrix_ADVI)
# y_predicted_test_ADVI = jnp.argmax(cat_probs_test_ADVI, 1)
# test_acc_ADVI = np.mean(np.array(y_predicted_test_ADVI == y_test))
# print(f"\nThe mean holdout accuracy using ADVI is {test_acc_ADVI:.03}")


###
# CAVI Inference
###

max_n_iterations_CAVI = 100

### fit the model with IB-CAVI
results_CAVI = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=max_n_iterations_CAVI,
)
df_performance_cavi = results_CAVI.holdout_performance_over_time
print(
    df_performance_cavi[
        [
            "seconds elapsed (cavi)",
            "mean holdout log likelihood for CBC_PROBIT",
            "mean holdout likelihood for CBC_PROBIT",
            "correct classification rate",
        ]
    ]
)

from categorical_from_binary.evaluate.multiclass import (
    Metric,
    evaluate_multiclass_regression_with_beta_estimate,
)


beta_mean_CAVI = results_CAVI.variational_params.beta.mean
print("\nCAVI log like on training set...")
evaluate_multiclass_regression_with_beta_estimate(
    covariates_train,
    labels_train,
    beta_mean_CAVI,
    link_for_category_probabilities=Link.CBC_PROBIT,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)
print("\nADVI log like on training set...")
evaluate_multiclass_regression_with_beta_estimate(
    covariates_train,
    labels_train,
    beta_mean_ADVI,
    link_for_category_probabilities=Link.CBC_PROBIT,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)
print("\nCAVI log like on test set...")
evaluate_multiclass_regression_with_beta_estimate(
    covariates_test,
    labels_test,
    beta_mean_CAVI,
    link_for_category_probabilities=Link.CBC_PROBIT,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)
print("\nADVI log like on test set...")
evaluate_multiclass_regression_with_beta_estimate(
    covariates_test,
    labels_test,
    beta_mean_ADVI,
    link_for_category_probabilities=Link.CBC_PROBIT,
    metric=Metric.MEAN_LOG_LIKELIHOOD,
)
