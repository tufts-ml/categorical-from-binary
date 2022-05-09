import glob

import pandas as pd
from pandas.core.frame import DataFrame


def make_df_from_directory_taking_one_file_for_each_seed(
    results_dir: str,
    prefix_for_filenames_of_interest: str = "sim_results_CAVI_vs_MLE_",
    verbose: bool = False,
) -> DataFrame:
    """
    Read separate files from disk and concatenate into a single dataframe.
    Each file from disk corresponds to results from a different seed, but contains
    results from many data contexts (K,M,N,sigma_high).

    The returned concatentated DataFrame has one row per seed x data context.

    The code assumes the filenames contain the string "seed={seed}_" in there somewhere.
    We only take one file from each seed, since sometimes on the cluster I launch multiple attempts
    simultaneously on different partitions (hugheslab, batch, ccpu, etc.), so there can be files
    with redundant information.
    """
    df_list = []
    seeds_used_so_far = []
    for filename in glob.glob(f"{results_dir}/{prefix_for_filenames_of_interest}*"):
        if verbose:
            print(f"Now scanning {filename}")
        seed = int(filename.split("seed=")[1].split("_")[0])
        if seed not in seeds_used_so_far:
            df_for_one_seed = pd.read_csv(filename)
            df_for_one_seed = df_for_one_seed.loc[
                :, ~df_for_one_seed.columns.str.contains("Unnamed")
            ]
            df_list.append(df_for_one_seed)
        seeds_used_so_far.append(seed)
    return pd.concat(df_list, ignore_index=True)
