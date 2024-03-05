import pandas as pd

name = '10K_accts'
path = f'~/UnreliableLabels/flib/AMLsim/outputs/{name}/tx_log.csv'

df = pd.read_csv(path)

print(df.head())
print(df.columns)

# print the unique values in the 'type' column
print(df['type'].unique())

# Count the values for isSAR
print(df['isSAR'].value_counts())

# Create new dataframe where isSAR == 1 and export to csv
df_sar = df[df['isSAR'] == 0]
# df_sar.to_csv(f'~/UnreliableLabels/test/{name}_normal.csv', index=False)

# Export the first 10000 rows to csv
df_sar = df_sar.iloc[:20000]   
df_sar.to_csv(f'~/UnreliableLabels/test/{name}_normal_10K.csv', index=False)