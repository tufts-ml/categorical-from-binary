"""
Here we compare two estimators (CBC probit and CBM Probit) for the category probabilities 
given beta-hat when 
(a) we use the variational posterior expectation for beta-hat
(b) the dataset has covariates
"""
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.evaluate.multiclass import (
    SplitDataset,
    take_measurements_comparing_CBM_and_CBC_estimators,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


def test_take_measurements_comparing_CBM_and_CBC_estimators():

    ###
    # Construct dataset
    ###
    seed = 1
    n_categories = 2
    n_features = 2
    n_samples = 50
    n_train_samples = 40
    include_intercept = True
    link_for_generating_data = Link.CBC_PROBIT
    dataset = generate_multiclass_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        beta_0=None,
        link=link_for_generating_data,
        seed=seed,
        include_intercept=include_intercept,
    )

    # Prep training data
    covariates_train = dataset.features[:n_train_samples]
    labels_train = dataset.labels[:n_train_samples]
    # `labels_train` gives one-hot encoded representation of category

    ####
    # Variational Inference
    ####
    convergence_criterion_drop_in_mean_elbo = 1
    results = compute_multiclass_probit_vi_with_normal_prior(
        labels_train,
        covariates_train,
        variational_params_init=None,
        convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
        max_n_iterations=2,
    )
    beta_mean = results.variational_params.beta.mean

    ###
    # Evaluate the model quality
    ###

    covariates_test = dataset.features[n_train_samples:]
    labels_test = dataset.labels[n_train_samples:]

    split_dataset = SplitDataset(
        covariates_train, labels_train, covariates_test, labels_test
    )
    measurements = take_measurements_comparing_CBM_and_CBC_estimators(
        split_dataset,
        beta_mean,
        link_for_generating_data,
        beta_ground_truth=dataset.beta,
    )
    # TODO: test that the non-value fields for measurements all have unique values.
    assert measurements is not None
