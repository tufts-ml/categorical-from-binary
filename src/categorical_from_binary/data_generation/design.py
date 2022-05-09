from typing import Union


class Design:
    """
    Attributes:
        num_exogenous : int
        num_intercept : (0 or 1 )
            0 if the design matrix does not include a column of all 1's for the intercept
            1 if it does
        num_autoregressive : (0 or K, the number of categories)
            0 if we are not using a (first-order) autoregression
            K if we are

    We assume that objects, such as regression weights, as blocked as follows
        [optional intercept term,  exogenous terms,  autoregressive terms]

    Usage:
        design = Design(n_features_exogenous, include_intercept, n_categories*is_autoregressive)
    """

    def __init__(
        self,
        num_exogenous: int,
        num_intercept: Union[bool, int] = 1,
        num_autoregressive: Union[bool, int] = 0,
    ):
        self.num_features_intercept = int(num_intercept)
        self.num_features_exogenous = int(num_exogenous)
        self.num_features_autoregressive = int(num_autoregressive)

    # TODO: Come up with a constructive name for features that don't include autoregressive features.

    @property
    def num_features_designed(self):
        return (
            self.num_features_intercept
            + self.num_features_exogenous
            + self.num_features_autoregressive
        )

    @property
    def num_features_non_intercept(self):
        return self.num_features_exogenous + self.num_features_autoregressive

    @property
    def num_features_non_autoregressive(self):
        return self.num_features_intercept + self.num_features_exogenous

    @property
    def index_bounds_exogenous_features(self):
        return (self.num_features_intercept, self.num_features_non_autoregressive)


# Useage
# design = Design(n_features_exogenous, include_intercept, n_categories*is_autoregressive)
