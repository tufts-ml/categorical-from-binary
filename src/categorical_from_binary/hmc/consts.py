from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_cbc_logit_probabilities,
    construct_cbc_probit_probabilities,
    construct_cbm_logit_probabilities,
    construct_cbm_probit_probabilities,
    construct_non_identified_softmax_probabilities,
)
from categorical_from_binary.hmc.core import Link


CATEGORY_PROBABILITY_FUNCTION_BY_MODEL_TYPE = {
    Link.CBC_PROBIT: construct_cbc_probit_probabilities,
    Link.CBM_PROBIT: construct_cbm_probit_probabilities,
    Link.CBC_LOGIT: construct_cbc_logit_probabilities,
    Link.CBM_LOGIT: construct_cbm_logit_probabilities,
    Link.SOFTMAX: construct_non_identified_softmax_probabilities,
}
