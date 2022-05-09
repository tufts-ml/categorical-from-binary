###
# Make heatmap plot
###

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


matplotlib.rc_file_defaults()


### now on local machine!!
path_to_mean_log_likes_df = "/Users/mwojno01/Repos/categorical_from_binary/data/results/cyber/mean_log_likes_from_fifty_user_round_robin.csv"
mean_log_likes_df = pd.read_csv(path_to_mean_log_likes_df, index_col=0)
user_domains = mean_log_likes_df.columns


### configs
values = np.array(mean_log_likes_df.values, dtype=float)
matrix_to_show = values >= np.diag(values)


### body of function

N_users = len(matrix_to_show)

cmap = matplotlib.cm.viridis
fig, ax = plt.subplots(1, 1, figsize=(6, 6), squeeze=True, constrained_layout=True)
# bounds = np.round(np.sort(percentiles + [0]), 3)
# norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# im = ax.imshow(matrix_to_show, cmap=cmap, norm=norm)
im = ax.imshow(matrix_to_show, cmap=cmap)
# plt.xticks(
#     np.arange(len(user_domains)),
#     user_domains,
#     rotation=80,
#     #    fontsize=x_ticks_fontsize,
# )
# plt.yticks(np.arange(len(user_domains)), user_domains)
plt.xlabel("Model owner", size=30)
plt.ylabel("Data owner", size=30)
# use one-indexing for tick labels
benchmarks = [1, 5, 10, 15, 20, 25, 30]
ax.set_xticks(benchmarks)
ax.set_xticklabels(benchmarks, size=24)
ax.set_yticks(benchmarks)
ax.set_yticklabels(benchmarks, size=24)
### try wrapping long names
# f = lambda x: textwrap.fill(x.get_text(), 30)
# ax.set_xticklabels(map(f, ax.get_xticklabels()))
# fig.colorbar(im, ax=ax)
plt.show()
