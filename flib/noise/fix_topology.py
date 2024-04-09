import pandas as pd

ratio = 0.25

name = '100K_accts_EASY25'
path = f'outputs/{name}/noise_log_Topology_full.csv'

df = pd.read_csv(path)

# In the df for every differnt type of "Reason", sample the ratio of the data
df = df.groupby('REASON').apply(lambda x: x.sample(frac=ratio))

df.to_csv(f'outputs/{name}/noise_log_Topology_{str(ratio)}.csv', index=False)