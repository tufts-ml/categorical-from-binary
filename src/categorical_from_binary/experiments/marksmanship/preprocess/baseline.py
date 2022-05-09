import glob
import os

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


DIR_WITH_BASELINE_CSVS = (
    "/Users/mwojno01/Box/IRB_Approval_Required/MASTR_E_Program_Data/Baseline_Data/"
)


def make_baseline_df(dir_with_baseline_csvs: str = DIR_WITH_BASELINE_CSVS) -> DataFrame:
    """
    Takes directory with a bunch of baseline .csv files (separate files for each baseline measure)
    and concatenates them into a single pandas DataFrame

    Acknowledgement:
        This function largely stolen from code written by Preetish Rath.
    """

    csv_files = glob.glob(os.path.join(dir_with_baseline_csvs, "*.csv"))

    # get the names of all the tables
    keys = []
    for csv_file in csv_files:
        keys.append(csv_file.split("/")[-1].replace(".csv", "").replace("19-022_", ""))

    # get all the baseline data into a single dataframe
    key_cols = ["Participant_ID"]
    ignore_cols = [
        "Time",
        "Notes",
        "MASTRE_ID",
        "MASTRE_Protocol_ID",
        "Protocol_ID",
        "Date",
        "NOTES",
        "1st Assistant",
        "BOOKNUM",
        "TODAYSDATE",
    ]
    all_cols = []
    total_cols = 0
    df_baseline = pd.DataFrame({})
    for i, key in enumerate(keys):
        csv_name = glob.glob(os.path.join(dir_with_baseline_csvs, "*" + key + "*.csv"))[
            0
        ]
        print(f"Now processing {csv_name}.")
        cur_df = pd.read_csv(csv_name)
        cur_df = cur_df.loc[:, ~cur_df.columns.duplicated()].copy()
        if "Participant ID" in cur_df.columns:
            cur_df = cur_df.rename(columns={"Participant ID": "Participant_ID"})

        if i > 0:
            drop_cols = list(set(cur_df.columns) & set(df_baseline.columns))
            # drop_cols = list(set(cur_df.columns))
            drop_cols = [col for col in drop_cols if col not in key_cols]
            cur_df = cur_df.drop(columns=drop_cols)

        numeric_cols = list(
            cur_df.columns[(cur_df.dtypes == "float64") | (cur_df.dtypes == "int64")]
        )
        numeric_cols = [
            col
            for col in numeric_cols
            if (col not in key_cols) & (col not in ignore_cols)
        ]
        all_cols.append(cur_df.columns)

        # parse non-numeric columns to categorical
        non_numeric_cols = []
        nan_inds = cur_df.isna()
        for col in cur_df.columns:
            if (
                (col not in key_cols)
                & (col not in numeric_cols)
                & (col not in ignore_cols)
            ):
                non_numeric_cols.append(col)
                cur_df[col] = cur_df[col].astype("category").cat.codes.astype("int64")
        cur_df[nan_inds] = np.nan
        total_cols += len(numeric_cols) + len(non_numeric_cols)
        print(
            "Number of columns extracted from %s : %d"
            % (key, len(numeric_cols) + len(non_numeric_cols))
        )
        print("Total  columns : %d" % (total_cols + 1))

        curr_cols = numeric_cols + non_numeric_cols
        if i == 0:
            df_baseline = cur_df[key_cols + curr_cols].copy()

        else:
            df_baseline = pd.merge(
                df_baseline, cur_df[key_cols + curr_cols], on=key_cols, how="outer"
            )

    return df_baseline
