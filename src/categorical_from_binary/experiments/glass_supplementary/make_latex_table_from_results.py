import numpy as np
import pandas as pd


# `PATH_TO_GLASS_IDENTIFICATION_RESULTS` should be the path to which the module
# `experiments.glass_supplementary.main.run_many_inference_strategies` wrote results.
PATH_TO_GLASS_IDENTIFICATION_RESULTS = "/Users/mwojno01/Repos/categorical_from_binary/data/results/glass_identification/glass_identification_results.csv"
df = pd.read_csv(PATH_TO_GLASS_IDENTIFICATION_RESULTS)

# We make a table which uses hierarchical index names
# Reference: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html

arrays = [
    [
        "Softmax",
        "CBT-Logit",
        "CBT-Logit",
        "CBM-Logit",
        "CBM-Logit",
        "CBT-Probit",
        "CBT-Probit",
        "CBM-Probit",
        "CBM-Probit",
    ],
    ["HMC"] + ["HMC", "IB-CAVI"] * 4,
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["Model", "Inference"])

# Now we rearrange the table.  Note that as of now the results were stored using OLD names for the models.

mean_likelihoods = pd.Series(
    [
        np.exp(df["holdout_loglike_softmax_HMC"].mean()),
        np.exp(df["holdout_loglike_CBC_Logit_HMC"].mean()),
        np.exp(df["holdout_loglike_IB_Logit_plus_CBC"].mean()),
        np.exp(df["holdout_loglike_CBM_Logit_HMC"].mean()),
        np.exp(df["holdout_loglike_IB_Logit_plus_CBM"].mean()),
        np.exp(df["holdout_loglike_CBC_Probit_HMC"].mean()),
        np.exp(df["holdout_loglike_IB_Probit_plus_CBC"].mean()),
        np.exp(df["holdout_loglike_CBM_Probit_HMC"].mean()),
        np.exp(df["holdout_loglike_IB_Probit_plus_CBM"].mean()),
    ],
    index=index,
)

accs = pd.Series(
    [
        df["accuracy_softmax_HMC"].mean(),
        df["accuracy_CBC_logit_HMC"].mean(),
        df["accuracy_IB_Logit_CAVI"].mean(),
        df["accuracy_CBM_logit_HMC"].mean(),
        df["accuracy_IB_Logit_CAVI"].mean(),
        df["accuracy_CBC_probit_HMC"].mean(),
        df["accuracy_IB_Probit_CAVI"].mean(),
        df["accuracy_CBM_probit_HMC"].mean(),
        df["accuracy_IB_Probit_CAVI"].mean(),
    ],
    index=index,
)

times = pd.Series(
    [
        df["time_for_softmax_HMC"].mean(),
        df["time_for_CBC_Logit_HMC"].mean(),
        df["time_for_IB_Logit"].mean(),
        df["time_for_CBM_Logit_HMC"].mean(),
        df["time_for_IB_Logit"].mean(),
        df["time_for_CBC_Probit_HMC"].mean(),
        df["time_for_IB_Probit"].mean(),
        df["time_for_CBM_Probit_HMC"].mean(),
        df["time_for_IB_Probit"].mean(),
    ],
    index=index,
)

dff = pd.DataFrame(
    [mean_likelihoods, accs, times],
    index=["Mean likelihood", "Accuracy", "Computation time"],
)
dff.round(decimals=2)
print(dff.to_latex(float_format="%.2f", index=True))
