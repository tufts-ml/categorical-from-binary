import numpy as np
from pandas.core.frame import DataFrame


np.set_printoptions(precision=3, suppress=True)


DATA_COLUMNS_TO_USE = [
    "TP",
    "Platoon",
    "Aiming_time",
    "Rotation",
    "Session",
    "Direction",
    "FP",
    "hit",
    "com_hit",
]


def preprocess_df_dynamic(df_dynamic_full: DataFrame) -> DataFrame:
    """
    Reduce dynamic dataframe to desired covariates
    Remove na's or rows with weird values
    Rename "TP" column as "Participant_Info" to align with baseline covariates
    """

    df_small = df_dynamic_full[DATA_COLUMNS_TO_USE]
    df = df_small.dropna()
    df = df[df["Aiming_time"] != 0]
    return df.rename(columns={"TP": "Participant_ID"})
