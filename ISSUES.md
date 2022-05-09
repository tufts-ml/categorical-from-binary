
# `categorical_from_binary`

* `categorical_from_binary.data_generation.hierarchical_multiclass_reg` has much greater flexibility
than `categorical_from_binary.data_generation.bayes_multiclass_reg`.  However, there is no reason to not percolate
this flexibility downward; just haven't gotten to it yet.  The additional flexibility consist of things like
using autoregressions, allowing control over the mean and variance of the different beta blocks (intercept, 
exogenous, autoregressive)

