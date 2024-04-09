import pandas as pd
import os

names = ['100K_accts_EASY25', '100K_accts_HARD1', '100K_accts_MID5']
ratios = [0.1, 0.25]

renames = {'noise_log_Neighbour_0.1': 'noise_neighbour.csv',
           'noise_log_Neighbour_0.25': 'noise_neighbour.csv',
           'noise_log_Topology - bipartite_0.1': 'noise_bipartite.csv',
           'noise_log_Topology - bipartite_0.25': 'noise_bipartite.csv',
           'noise_log_Topology - fan_in_0.1': 'noise_fan_in.csv',
           'noise_log_Topology - fan_in_0.25': 'noise_fan_in.csv',
           'noise_log_Topology - fan_out_0.1': 'noise_fan_out.csv',
           'noise_log_Topology - fan_out_0.25': 'noise_fan_out.csv',
           'noise_log_Topology - cycle_0.1': 'noise_cycle.csv',
           'noise_log_Topology - cycle_0.25': 'noise_cycle.csv',
           'noise_log_Topology - scatter_gather_0.1': 'noise_scatter_gather.csv',
           'noise_log_Topology - scatter_gather_0.25': 'noise_scatter_gather.csv',
           'noise_log_Topology - gather_scatter_0.1': 'noise_gather_scatter.csv',
           'noise_log_Random_FN_0.1': 'noise_FN.csv',
           'noise_log_Random_FN_0.25': 'noise_FN.csv',
           'noise_log_Random_FP_0.1': 'noise_FP.csv',
           'noise_log_Random_FP_0.25': 'noise_FP.csv',}

# To all keys add .csv
renames = {key + '.csv': value for key, value in renames.items()}

for name in names:
    for ratio in ratios:
        path = f'../noise/{name}/{ratio}/'
        for file in os.listdir(path):
            if file in renames:

                df = pd.read_csv(path + file)
                df.to_csv(path + renames[file], index=False)
                os.remove(path + file)
                print(f'{file} renamed to {renames[file]}')
        