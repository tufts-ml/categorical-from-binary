# this will need to happen before restircitng to a single user,
# and would get passed.  if not passed, then we could make a user specific
# dictionary, but note then we'd have to have an "other" category.  perhaps we should
# add that in anywasys

import numpy as np
from bidict import bidict
from pandas.core.frame import DataFrame
from pandas.core.series import Series


def construct_minimal_process_id_by_process_id(df: DataFrame) -> bidict:
    """
    There are 24,742 unique process names in the human process start data frame.
    But the max integer value seen in the representation used is 62,985.
    We may not  want to deal with 62,985 categories if they don't exist, so we make a bidirectional
    dictionary allowing us to bijectively map
    the original representation (in {1,...,62985}) to a minimal
    (and zero-indexed) representation (in {0, ..., 24741}).

    Arguments:
        df: DataFrame, the return value of `load_human_process_start_df`
    Returns:
        bidict whose keys are the original process identifiers and whose values are the minimal process identifiers.
        since it's a bidirectional dictionary, we can go backwards by just calling d.inverse on the dict.
    """

    process_names = set(df["process name"])
    label_ints = [int(x[1:]) for x in process_names]
    label_ints_ascending = np.sort(label_ints)

    minimal_process_id_by_process_id = bidict()
    for idx, val in enumerate(label_ints_ascending):
        minimal_process_id_by_process_id[val] = idx
    return minimal_process_id_by_process_id


def compute_num_of_unique_processes_from_dict_of_minimal_process_id_by_process_id(
    minimal_process_id_by_process_id: bidict,
):
    # To compute the number of unique processes, we must add one to the maximum value to account for zero-indexing
    return max(minimal_process_id_by_process_id.values()) + 1


def compute_sorted_sample_sizes_for_users(df: DataFrame) -> Series:
    """
    Sorts users by size (in descending order).

    Arguments:
        df: DataFrame, the return value of `load_human_process_start_df`

    Returns:
        A pandas Series giving the number of observations (i.e. process start events) for each user-domain
        The index is the user-domain, and the value is the number of observations.

        Sample Return:
            user@domain
            U177@DOM1       996702
            U5356@DOM1      679785
            U3083@DOM1      562502
            U3665@DOM1      508014
            U1345@DOM1      443278
                            ...
            U8946@C16633         1
            U1314@C24042         1
            U10161@C812          1
            U5009@C1042          1
            U8816@C18727         1
            Name: time, Length: 10620, dtype: int64

    """
    sample_sizes_for_users = df.groupby(by="user@domain")["time"].agg("count")
    return sample_sizes_for_users.sort_values(ascending=False)
