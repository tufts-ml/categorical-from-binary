from typing import Optional

import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.experiments.marksmanship.preprocess.baseline import (
    make_baseline_df,
)
from categorical_from_binary.experiments.marksmanship.preprocess.dynamic import (
    preprocess_df_dynamic,
)
from categorical_from_binary.experiments.marksmanship.preprocess.util import (
    exclude_based_on_missingness,
)


def join_dynamic_and_baseline_dataframes(
    df_dynamic: DataFrame, df_baseline: DataFrame
) -> DataFrame:
    return df_dynamic.merge(df_baseline, how="left", on="Participant_ID")


DIR_WITH_BASELINE_CSVS = (
    "/Users/mwojno01/Box/IRB_Approval_Required/MASTR_E_Program_Data/Baseline_Data/"
)
PATH_TO_DYNAMIC_DATA = "/Users/mwojno01/Research/ResearchArtifacts/MASTR-E/marksmanship/data/ISS_72HFS_06162021_All[1964].xlsx"


def make_marksmanship_data_frame(
    use_baseline_covariates: bool,
    threshold_on_proportion_missing_in_columns_for_baseline_data: Optional[
        float
    ] = None,
) -> DataFrame:

    if use_baseline_covariates:
        raise NotImplementedError(
            "The code as is will not return a meaningful dataset."
        )

    # dynamic covariates
    df_dynamic_full = pd.read_excel(PATH_TO_DYNAMIC_DATA)
    df_dynamic_preprocessed = preprocess_df_dynamic(df_dynamic_full)

    # merge the static and dynamic covariates
    if use_baseline_covariates:

        # baseline covariates
        # df_baseline = make_baseline_df(DIR_WITH_BASELINE_CSVS)
        df_baseline_full = make_baseline_df(DIR_WITH_BASELINE_CSVS)
        df_baseline = exclude_based_on_missingness(
            df_baseline_full,
            threshold_on_proportion_missing_in_columns_for_baseline_data,
        )

        return join_dynamic_and_baseline_dataframes(
            df_dynamic_preprocessed, df_baseline
        )
    else:
        return df_dynamic_preprocessed
