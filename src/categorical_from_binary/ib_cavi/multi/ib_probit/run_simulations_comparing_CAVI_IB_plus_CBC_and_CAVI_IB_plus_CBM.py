"""
Here we compare two estimators (CBC probit and CBM Probit) for the category probabilities 
given beta-hat when 
(a) we use the variational posterior expectation for beta-hat
(b) the dataset has covariates
"""

import datetime
import os
import pickle

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.evaluate.multiclass import (
    take_measurements_comparing_CBM_and_CBC_estimators,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.results import ResultsOnOneSplit


###
# Outer layer for running many simulations.
# For a demo, set n_simulations = 1
###
n_simulations_per_link_for_generating_data = 10
measurements_on_all_splits = []
results_on_many_splits = []

for link_for_generating_data in [
    Link.MULTI_PROBIT,
    Link.MULTI_LOGIT,
    Link.CBC_PROBIT,
    Link.CBM_PROBIT,
]:
    for i in range(n_simulations_per_link_for_generating_data):
        print(
            f"\n---- Now running simulation {i+1}/{n_simulations_per_link_for_generating_data} for "
            f"data generated via link {link_for_generating_data}----"
        )

        ###
        # Construct dataset
        ###
        seed = i
        n_categories = 3
        n_features = 3
        n_samples = 5000
        n_train_samples = 4000
        include_intercept = True
        link_for_generating_data = link_for_generating_data
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
        convergence_criterion_drop_in_elbo = 1.0
        results = compute_multiclass_probit_vi_with_normal_prior(
            labels_train,
            covariates_train,
            variational_params_init=None,
            convergence_criterion_drop_in_elbo=convergence_criterion_drop_in_elbo,
        )
        variational_params = results.variational_params
        beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov

        ###
        # Evaluate the model quality
        ###

        covariates_test = dataset.features[n_train_samples:]
        labels_test = dataset.labels[n_train_samples:]

        split_dataset = SplitDataset(
            covariates_train, labels_train, covariates_test, labels_test
        )
        results_on_one_split = ResultsOnOneSplit(
            i,
            split_dataset,
            beta_mean=beta_mean,
            beta_cov=beta_cov,
            link_for_generating_data=link_for_generating_data,
        )
        results_on_many_splits.append(results_on_one_split)

        measurements_on_one_dataset = (
            take_measurements_comparing_CBM_and_CBC_estimators(
                split_dataset,
                beta_mean,
                link_for_generating_data,
                beta_ground_truth=dataset.beta,
            )
        )
        measurements_on_all_splits.extend(measurements_on_one_dataset)

###
# SAVING TO DISK
###

save_to_disk = False
save_dir = "data/cbc_probit/simulations/"

if save_to_disk:
    todays_date = str(datetime.datetime.now().date())
    results_basename = (
        "simulation_results_comparing_IB_plus_CBM_to_IB_plus_CBC_for_different_data_generating_links_"
        + todays_date
        + ".pkl"
    )
    path_to_results = os.path.join(save_dir, results_basename)
    with open(path_to_results, "wb") as handle:
        pickle.dump(results_on_many_splits, handle, protocol=pickle.HIGHEST_PROTOCOL)
