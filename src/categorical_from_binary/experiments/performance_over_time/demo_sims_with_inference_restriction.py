from categorical_from_binary.performance_over_time.classes import InferenceType
from categorical_from_binary.performance_over_time.main import run_performance_over_time


path_to_configs = "configs/performance_over_time/demo_sims.yaml"
run_performance_over_time(
    path_to_configs, only_run_this_inference=InferenceType.CAVI_PROBIT, make_plots=False
)
run_performance_over_time(
    path_to_configs, only_run_this_inference=InferenceType.CAVI_LOGIT, make_plots=False
)
