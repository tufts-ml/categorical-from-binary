import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


np.set_printoptions(precision=3, suppress=True)
import typing
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from categorical_from_binary.covariate_dict import VariableInfo, VariableType
from categorical_from_binary.experiments.marksmanship.preprocess.util import (
    standardize_numpy_matrix_by_columns,
)
from categorical_from_binary.types import NumpyArray2D


def make_labels_and_label_dict_from_df(
    df: DataFrame,
) -> Tuple[NumpyArray2D, typing.OrderedDict[str, VariableInfo]]:
    """
    Returns:
        tuple of labels and label dict
        labels:
            One-hot encoded numpy array with shape (N,K), where N is number of samples
            and K is number of categories (labels)

            Labels:
                0: Shoot at non_threat
                1: Miss threat
                2: Hit threat but not COM
                3: COM hit to threat
    """
    shoot_at_non_threat = (df["FP"]).to_numpy(dtype=int)
    miss_threat = ((1 - df["hit"]) * (1 - df["com_hit"]) * (1 - df["FP"])).to_numpy(
        dtype=int
    )
    hit_threat_outside_COM = (
        df["hit"] * (1 - df["com_hit"]) * (1 - df["FP"])
    ).to_numpy(dtype=int)
    hit_threat_in_COM = (df["hit"] * (df["com_hit"]) * (1 - df["FP"])).to_numpy(
        dtype=int
    )
    labels = np.vstack(
        (shoot_at_non_threat, miss_threat, hit_threat_outside_COM, hit_threat_in_COM)
    ).T
    label_dict = OrderedDict(
        [
            ("shoot at non-threat", VariableInfo(VariableType.BINARY, index=0)),
            ("miss threat", VariableInfo(VariableType.BINARY, index=1)),
            ("hit threat at periphery", VariableInfo(VariableType.BINARY, index=2)),
            ("hit threat at center", VariableInfo(VariableType.BINARY, index=3)),
        ]
    )
    return labels, label_dict


def make_covariates_and_covariate_dict_from_df(
    df, standardize=True
) -> Tuple[NumpyArray2D, Dict[str, VariableInfo]]:
    """
    Returns: tuple of covariates and feature dict
        covariates:
            numpy array of shape (N,M+1),
            where N is the number of samples and M+1 is the number of covariates INCLUDING intercept.
        covariate dict:
            dict mapping feature name to covariate info
    """
    covariate_dict = {}
    n = np.shape(df)[0]
    ones_vector = np.ones(n)[:, np.newaxis]

    # we go through columns one at a time, and manually transform some;
    # the others we leave as is.
    candidate_covariates = df.columns.to_list()
    candidate_covariates.remove("Participant_ID")  # we remove these outright
    candidate_covariates.remove("FP")
    candidate_covariates.remove("hit")
    candidate_covariates.remove("com_hit")

    abs_rotation = np.abs(df["Rotation"]).to_numpy()
    candidate_covariates.remove("Rotation")

    log_aiming_time = np.log(df["Aiming_time"]).to_numpy()
    candidate_covariates.remove("Aiming_time")

    session = pd.get_dummies(df["Session"], prefix="Session").to_numpy()[:, :-1]
    candidate_covariates.remove("Session")

    direction = pd.get_dummies(df["Direction"], prefix="Direction").to_numpy()[:, :-1]
    candidate_covariates.remove("Direction")

    platoon = pd.get_dummies(df["Platoon"], prefix="Platoon").to_numpy()[:, 1:]
    candidate_covariates.remove("Platoon")

    dynamic_covariates_no_intercept = np.vstack(
        (log_aiming_time, session.T, abs_rotation, direction.T, platoon.T)
    ).T
    baseline_covariates_no_intercept = df[candidate_covariates].to_numpy()
    covariates_without_intercept = np.hstack(
        (dynamic_covariates_no_intercept, baseline_covariates_no_intercept)
    )

    if standardize:
        covariates_without_intercept_possibly_standardized = (
            standardize_numpy_matrix_by_columns(covariates_without_intercept)
        )

    covariates = np.hstack(
        (ones_vector, covariates_without_intercept_possibly_standardized)
    )

    # we add in np.nan at the beginning to account for intercept
    means = np.round(
        np.hstack(([np.nan], np.mean(covariates_without_intercept, axis=0))), 2
    )
    stds = np.round(
        np.hstack(([np.nan], np.std(covariates_without_intercept, axis=0))), 2
    )
    mins = np.round(
        np.hstack(([np.nan], np.min(covariates_without_intercept, axis=0))), 2
    )
    maxs = np.round(
        np.hstack(([np.nan], np.max(covariates_without_intercept, axis=0))), 2
    )

    covariate_dict_dynamic = {
        "intercept": VariableInfo(VariableType.CONTINUOUS, index=0),
        "Log aim time": VariableInfo(
            VariableType.CONTINUOUS,
            index=1,
            raw_mean=means[1],
            raw_std=stds[1],
            raw_min=mins[1],
            raw_max=maxs[1],
        ),
        "Session 2, not 1": VariableInfo(
            VariableType.BINARY,
            index=2,
            raw_mean=means[2],
            raw_std=stds[2],
            raw_min=mins[2],
            raw_max=maxs[2],
        ),
        "Session 3, not 1": VariableInfo(
            VariableType.BINARY,
            index=3,
            raw_mean=means[3],
            raw_std=stds[3],
            raw_min=mins[3],
            raw_max=maxs[3],
        ),
        "Absolute rotation": VariableInfo(
            VariableType.CONTINUOUS,
            index=4,
            raw_mean=means[4],
            raw_std=stds[4],
            raw_min=mins[4],
            raw_max=maxs[4],
        ),
        "Direction right, not left": VariableInfo(
            VariableType.BINARY,
            index=5,
            raw_mean=means[5],
            raw_std=stds[5],
            raw_min=mins[5],
            raw_max=maxs[5],
        ),
        "Platoon B, not A": VariableInfo(
            VariableType.BINARY,
            index=6,
            raw_mean=means[6],
            raw_std=stds[6],
            raw_min=mins[6],
            raw_max=maxs[6],
        ),
        "Platoon C, not A": VariableInfo(
            VariableType.BINARY,
            index=7,
            raw_mean=means[7],
            raw_std=stds[7],
            raw_min=mins[7],
            raw_max=maxs[7],
        ),
    }

    # assume all the remaining covariates are continuous...might be wrong
    covariate_dict_baseline = {}
    for i, covariate_name in enumerate(candidate_covariates):
        covariate_dict_baseline[covariate_name] = VariableInfo(
            VariableType.CONTINUOUS, index=8 + i
        )

    covariate_dict = {**covariate_dict_dynamic, **covariate_dict_baseline}

    return covariates, covariate_dict


def add_previous_label_to_covariates(
    covariates: NumpyArray2D,
    covariate_dict: Dict[str, VariableInfo],
    labels: NumpyArray2D,
    label_dict: Dict[str, VariableInfo],
    remove_intercept_when_adding_autoregressive_labels: bool,
    use_one_label_as_autoregressive_baseline_label: bool = False,
    autoregressive_baseline_label: Optional[str] = None,
    where_to_put_autoregressive_labels_after: str = "end",
) -> Tuple[
    NumpyArray2D, Dict[str, VariableInfo], NumpyArray2D, Dict[str, VariableInfo]
]:
    """
    Arguments:
        covariates: numpy array of shape (N,M+1),
            where N is the number of samples and M+1 is the number of covariates INCLUDING intercept.
        labels: One-hot encoded numpy array with shape (N,K), where N is number of samples
            and K is number of categories (labels)
        use_one_label_as_autoregressive_baseline_label: bool. If True, then that label is considered a
            baseline, and is removed from the covariates.
        where_to_put_autoregressive_labels_after  : str.  Must be one of "intercept" or "end".

    Returns:
        tuple of adjusted covariates and previous labels.  We don't include all the labels because we
        consider them adjustments to the intercept
    """
    return _add_label_n_steps_back_to_covariates(
        covariates,
        covariate_dict,
        labels,
        label_dict,
        n=1,
        remove_intercept_when_adding_autoregressive_labels=remove_intercept_when_adding_autoregressive_labels,
        use_one_label_as_autoregressive_baseline_label=use_one_label_as_autoregressive_baseline_label,
        autoregressive_baseline_label=autoregressive_baseline_label,
        where_to_put_autoregressive_labels_after=where_to_put_autoregressive_labels_after,
    )


def add_second_previous_label_to_covariates(
    covariates: NumpyArray2D,
    covariate_dict: Dict[str, VariableInfo],
    labels: NumpyArray2D,
    label_dict: Dict[str, VariableInfo],
    remove_intercept_when_adding_autoregressive_labels: bool,
    use_one_label_as_autoregressive_baseline_label: bool = False,
    autoregressive_baseline_label: Optional[str] = None,
    where_to_put_autoregressive_labels_after: str = "end",
) -> Tuple[
    NumpyArray2D, Dict[str, VariableInfo], NumpyArray2D, Dict[str, VariableInfo]
]:
    """
    Arguments:
        covariates: numpy array of shape (N,M+1),
            where N is the number of samples and M+1 is the number of covariates INCLUDING intercept.
        labels: One-hot encoded numpy array with shape (N,K), where N is number of samples
            and K is number of categories (labels)
        use_one_label_as_autoregressive_baseline_label: bool. If True, then that label is considered a
            baseline, and is removed from the covariates.
        where_to_put_autoregressive_labels_after  : str.  Must be one of "intercept" or "end".
    """
    return _add_label_n_steps_back_to_covariates(
        covariates,
        covariate_dict,
        labels,
        label_dict,
        n=2,
        remove_intercept_when_adding_autoregressive_labels=remove_intercept_when_adding_autoregressive_labels,
        use_one_label_as_autoregressive_baseline_label=use_one_label_as_autoregressive_baseline_label,
        autoregressive_baseline_label=autoregressive_baseline_label,
        where_to_put_autoregressive_labels_after=where_to_put_autoregressive_labels_after,
    )


def _add_label_n_steps_back_to_covariates(
    covariates: NumpyArray2D,
    covariate_dict: Dict[str, VariableInfo],
    labels: NumpyArray2D,
    label_dict: Dict[str, VariableInfo],
    n: int,
    remove_intercept_when_adding_autoregressive_labels: bool,
    use_one_label_as_autoregressive_baseline_label: bool = False,
    autoregressive_baseline_label: Optional[str] = None,
    where_to_put_autoregressive_labels_after: str = "end",
) -> Tuple[
    NumpyArray2D, Dict[str, VariableInfo], NumpyArray2D, Dict[str, VariableInfo]
]:
    """
    Arguments:
        covariates: numpy array of shape (N,M+1),
            where N is the number of samples and M+1 is the number of covariates INCLUDING intercept.
        labels: One-hot encoded numpy array with shape (N,K), where N is number of samples
            and K is number of categories (labels)
        use_one_label_as_autoregressive_baseline_label: bool. If True, then that label is considered a
            baseline, and is removed from the covariates.
        where_to_put_autoregressive_labels_after  : str.  Must be one of "intercept" or "end".
    """
    if (
        remove_intercept_when_adding_autoregressive_labels
        and use_one_label_as_autoregressive_baseline_label
    ):
        raise ValueError(
            "You can't remove BOTH the intercept AND an autoregressive baseline label when adding "
            "in the autoregressive information."
        )

    label_names = list(label_dict.keys())
    if use_one_label_as_autoregressive_baseline_label:
        autoregressive_baseline_label_index = label_names.index(
            autoregressive_baseline_label
        )
        label_names = [
            f"Prev. {x}, not {autoregressive_baseline_label} " for x in label_names
        ]
    label_as_feature_names = [x + f" {n} trials back" for x in label_names]

    # to make an autoregressive adjustment to the covariates, we must throw away the last $n$ labels and append those
    # the the first $n$ covariates
    previous_labels = labels[:-n]
    covariates_without_first_obs = covariates[n:]

    # possibly adjust previous labels based on a desired baseline label
    if use_one_label_as_autoregressive_baseline_label:
        previous_labels_possibly_excluding_autoregressive_baseline_label = np.delete(
            previous_labels, autoregressive_baseline_label_index, axis=1
        )
    else:
        previous_labels_possibly_excluding_autoregressive_baseline_label = (
            previous_labels
        )

    # now adjust the covariates be right after the intercept term.
    intercept_covariate_index = covariate_dict["intercept"].index
    largest_covariate_index = max([x.index for x in covariate_dict.values()])
    if where_to_put_autoregressive_labels_after == "intercept":
        splice_index = intercept_covariate_index
    elif where_to_put_autoregressive_labels_after == "end":
        splice_index = largest_covariate_index
    else:
        raise ValueError("I'm not sure where to put the autoregressive labels")

    covariates_adjusted = np.insert(
        covariates_without_first_obs,
        splice_index + 1,
        previous_labels_possibly_excluding_autoregressive_baseline_label.T,
        axis=1,
    )

    # now adjust covariate dict to add labels from earlier trial
    # TODO: all this would have been so much easier if we were working with a dataframe!
    if use_one_label_as_autoregressive_baseline_label:
        label_as_feature_names.pop(autoregressive_baseline_label_index)
    num_autoregressive_covariates = len(label_as_feature_names)

    if where_to_put_autoregressive_labels_after == "intercept":
        for covariate_name, covariate_info in covariate_dict.items():
            if covariate_name != "intercept":
                covariate_info.index += num_autoregressive_covariates

    for (i, label_as_feature_name) in enumerate(label_as_feature_names):
        covariate_dict.update(
            {
                label_as_feature_name: VariableInfo(
                    VariableType.BINARY, index=splice_index + i + 1
                )
            }
        )

    if remove_intercept_when_adding_autoregressive_labels:
        # adjust covariate dict
        del covariate_dict["intercept"]
        for covariate_name, covariate_info in covariate_dict.items():
            covariate_info.index -= 1

        # adjust covariates (again)
        if not (covariates_adjusted[:, 0] == 1).all():
            raise ValueError(
                "You asked me to remove the intercept term, but it doesn't exist!"
            )
        else:
            covariates_adjusted = np.delete(covariates_adjusted, 0, 1)
    return covariates_adjusted, covariate_dict, labels[n:]
