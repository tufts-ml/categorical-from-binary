import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.pandas_helpers import (
    filter_df_rows_by_column_value_starts,
    filter_df_rows_by_column_values,
)


def extract_human_process_starts_from_df(df_full: DataFrame) -> DataFrame:
    """
    Takes the full cybersecurity process starts dataframe, and retain only the process starts (not process ends),
    and retain only rows that correspond to human user activity (i.e. user@domain starts with `U`)
    """
    df_starts = filter_df_rows_by_column_values(
        df_full, col="Start or End", values="End"
    )
    return filter_df_rows_by_column_value_starts(
        df_starts, col="user@domain", start="U"
    )


def load_preprocessed_df_in_chunks(path: str, chunk_size: int = 100000) -> DataFrame:
    """
    Load cybersecurity data as dataframe in chunks and preprocess along the way

    Arguments:
        path: Path to the uncompressed version of proc.txt.gz located at https://csr.lanl.gov/data/cyber1.
        chunk_size: The number of lines to grab from the raw file at a time, before preprocessing
            and adding to a slowly growing DataFrame.

            For reference, the uncompressed version of proc.txt.gz referenced above has
            L=426,045,096 lines.   So one would expect processing to be complete after
            the number of chunks processed equals chunk_size/L.

            I have gotten good results with chunk_size being 1,000,000 (and so requiring 427 chunks).

    Returns:
        dataframe with dimensionality (N,E)
        where N is the number of process starts
        and E is the number of events, described in the form
        "time,user@domain,computer,process name,start or end”

        For more information, see https://csr.lanl.gov/data/cyber1/ proc.txt.gz
    """
    print("--------------------")
    print("Loading " + path)
    print("--------------------")
    df = pd.DataFrame()

    column_names = ["time", "user@domain", "computer", "process name", "Start or End"]
    chunk_num = 0
    for df_chunk_full in pd.read_csv(
        path,
        delimiter=",",
        names=column_names,
        iterator=True,
        chunksize=chunk_size,
    ):
        chunk_num += 1
        print(f"Now processing chunk {chunk_num}", end="\r")
        df_chunk = extract_human_process_starts_from_df(df_chunk_full)
        df = pd.concat([df, df_chunk])

    print("\nComplete")
    return df


def load_raw_process_df(path: str) -> DataFrame:
    """
    Load cybersecurity process start and end data as dataframe.  Note that there is not yet
    any filtering to process starts or human users.

    Arguments:
        path: Path to the uncompressed version of proc.txt.gz located at https://csr.lanl.gov/data/cyber1.

    Returns:
        dataframe with dimensionality (N,E)
        where N is the number of process starts
        and E is the number of events, described in the form
        "time,user@domain,computer,process name,start or end”

        For more information, see https://csr.lanl.gov/data/cyber1/ proc.txt.gz
    """
    print("--------------------")
    print("Loading " + path)
    print("--------------------")
    column_names = ["time", "user@domain", "computer", "process name", "Start or End"]
    data = pd.read_csv(path, delimiter=",", names=column_names)
    print("Complete")
    return data


def load_human_process_start_df(path):
    """
    Load the human process start dataframe, which was effectively created by
    running load_raw_process_df() on  the uncompressed version of proc.txt.gz
    located at https://csr.lanl.gov/data/cyber1, and then running extract_human_process_starts_from_df() on
    the result -- i.e.

        PATH_BIG =  "/cluster/tufts/hugheslab/mwojno01/data/proc_unzipped"  # 15 GB (426,045,096 lines)
        df_raw = load_raw_process_df(PATH_BIG)
        df = extract_human_process_starts_from_df(df_raw)

    In actuality, due to memory constraints, this dataframe was created by running
    load_preprocessed_df_in_chunks().  [The above codeblock might work (haven't tried it), but if not, one
    can always construct a smaller file via e.g. head -1000 /cluster/tufts/hugheslab/mwojno01/data/proc_unzipped > new_filename
    and then run the same code on this path.]
    """
    return pd.read_csv(
        path, delimiter=",", usecols=["time", "user@domain", "computer", "process name"]
    )
