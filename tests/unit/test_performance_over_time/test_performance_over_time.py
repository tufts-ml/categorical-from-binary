from categorical_from_binary.performance_over_time.main import run_performance_over_time


def test_that_performance_over_time_runs_on_demo_sims_without_crashing():
    path_to_configs = "configs/performance_over_time/demo_sims.yaml"
    run_performance_over_time(path_to_configs)
