import pandas as pd

ratio = 0.1

name = '100K_accts_HARD1'
path = f'../noise/{name}/{ratio}/noise_log_Topology_{ratio}.csv'

df = pd.read_csv(path)

# In the df split the csv file into several based on unique values of 'REASON'
for reason in df['REASON'].unique():
    df[df['REASON'] == reason].to_csv(f'../noise/{name}/{ratio}/noise_log_{reason}_{ratio}.csv', index=False)
