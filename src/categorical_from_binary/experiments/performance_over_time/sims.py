from categorical_from_binary.performance_over_time.main import run_performance_over_time


# `path_to_configs` can choose other size configs as well.
path_to_configs = "configs/performance_over_time/small_sims.yaml"
run_performance_over_time(path_to_configs)
