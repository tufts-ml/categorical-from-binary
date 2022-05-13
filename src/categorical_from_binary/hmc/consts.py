from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_cbc_logit_probabilities,
    construct_cbc_probit_probabilities,
    construct_cbm_logit_probabilities,
    construct_cbm_probit_probabilities,
    construct_non_identified_softmax_probabilities,
)
from categorical_from_binary.hmc.core import CategoricalModelType


CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE = {
    CategoricalModelType.CBC_PROBIT: construct_cbc_probit_probabilities,
    CategoricalModelType.CBM_PROBIT: construct_cbm_probit_probabilities,
    CategoricalModelType.CBC_LOGIT: construct_cbc_logit_probabilities,
    CategoricalModelType.CBM_LOGIT: construct_cbm_logit_probabilities,
    CategoricalModelType.SOFTMAX: construct_non_identified_softmax_probabilities,
}
