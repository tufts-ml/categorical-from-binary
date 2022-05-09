from categorical_from_binary.datasets.cyber.data_frame import load_human_process_start_df
from categorical_from_binary.pandas_helpers import keep_df_rows_by_column_values


### Load all data
PATH_HUMAN_STARTS = (
    "/cluster/tufts/hugheslab/mwojno01/data/human_process_starts.csv"  # 1.5 GB
)
df = load_human_process_start_df(PATH_HUMAN_STARTS)

### Show sample data
# show process starts , in temporal order,  for each of `n_users` users
n_users = 10
user_domains = list(set(df["user@domain"]))
if n_users is not None:
    user_domains_to_use = user_domains[:n_users]
else:
    user_domains_to_use = user_domains
for user_domain in user_domains_to_use:
    df_user_domain = keep_df_rows_by_column_values(
        df, col="user@domain", values=user_domain
    )
    print(df_user_domain, end="\n")

    # show the number of observations for each user
    # print(f"user_domain: {user_domain}, N: {len(df_user_domain)}")
