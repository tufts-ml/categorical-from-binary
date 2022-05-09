from categorical_from_binary.ib_cavi.multi.ib_probit.simulations.core.warping import (
    run_warping_simulations_on_softmax_data,
)


Ks = [3, 9, 27]
Ms = [1, 3, 9, 27]
Ns = [100, 1000, 10000]
n_datasets_per_data_dimension = 3

df = run_warping_simulations_on_softmax_data(
    Ks,
    Ns,
    Ms,
    n_datasets_per_data_dimension,
    run_cavi=True,
    convergence_criterion_drop_in_mean_elbo=0.01,
    run_sdo_and_do_mle_only_if_num_coefficients_is_less_than_this=200,
    # max_n_cavi_iterations=25,
    # run_cavi_only_for_one_seed=True,
    # convergence_criterion_drop_in_mean_elbo=0.01,
)
