import numpy as np
import pandas as pd
import seaborn as sns


# sns.set_theme(style="whitegrid")
sns.set(style="darkgrid")
import matplotlib.pyplot as plt


# PLOT 1

n_groups = 10
flat_pcts = [0.555, 0.637, 0.629, 0.568, 0.565, 0.498, 0.736, 0.646, 0.58, 0.576]
hierarchical_pcts = [
    0.585,
    0.657,
    0.642,
    0.594,
    0.584,
    0.526,
    0.762,
    0.679,
    0.621,
    0.604,
]

flat_pcts_sorted = [flat_pcts[i] for i in np.argsort(flat_pcts)]
hierarchical_pcts_sorted_the_same_way = [
    hierarchical_pcts[i] for i in np.argsort(flat_pcts)
]

choice_probs = flat_pcts_sorted + hierarchical_pcts_sorted_the_same_way
model_type = ["flat"] * n_groups + ["hierarchical"] * n_groups
group_ids = [i + 1 for i in range(n_groups)] * 2

df = pd.DataFrame(
    {
        "group": pd.Series(group_ids),
        "choice prob": pd.Series(choice_probs),
        "model_type": pd.Series(model_type),
    }
)

ax = sns.barplot(x="group", y="choice prob", hue="model_type", data=df)


plt.title("Mean choice probabilities by time series")
plt.show()

# PLOT 2
n_groups = 10
flat_pcts = [0.388, 0.288, 0.194, 0.0659, 0.482, 0.192, 0.092, 0.444, 0.431, 0.264]
hierarchical_pcts = [0.504, 0.503, 0.47, 0.674, 0.59, 0.403, 0.525, 0.706, 0.674, 0.585]
diff_in_pcts = np.array(hierarchical_pcts) - np.array(flat_pcts)


hierarchical_pcts_sorted = [hierarchical_pcts[i] for i in np.argsort(diff_in_pcts)]
flat_pcts_sorted = [flat_pcts[i] for i in np.argsort(diff_in_pcts)]

choice_probs = flat_pcts_sorted + hierarchical_pcts_sorted
# choice_probs =  flat_pcts + hierarchical_pcts
model_type = ["flat"] * n_groups + ["hierarch"] * n_groups
group_ids = [i + 1 for i in range(n_groups)] * 2

df = pd.DataFrame(
    {
        "sequence id": pd.Series(group_ids),
        "probability": pd.Series(choice_probs),
        "model_type": pd.Series(model_type),
    }
)

ax = sns.barplot(x="sequence id", y="probability", hue="model_type", data=df)
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# we can supply a rectangle box that the whole subplots area (including legend) will fit into
# plt.tight_layout(rect=[0, 0, 0.8, 0.95])
# plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.title("Mean probability across occurrences of the RAREST overall category")
plt.legend(loc="upper left", frameon=False)
plt.show()
