import pandas as pd
import numpy as np

name = '10K_accts'
path = f'~/UnreliableLabels/flib/AMLsim/outputs/{name}/tx_log.csv'

df = pd.read_csv(path)

# add new column to df 'isSARtampered' and set all values to 0

df['tamerped'] = 0

# Change with a chance of 0.1 the value of isSAR to 1 and set tampered to 1 as well

df.loc[df['isSAR'] == 1, 'tampered'] = np.random.choice([0,1], size=(len(df[df['isSAR'] == 1]),), p=[0.9999, 0.0001])