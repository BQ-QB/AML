import pandas

def apply_noise(data,noise_list):
    # For data add column 'true_label' as a copy of 'is_sar'
    data['true_label'] = data['is_sar']
    noise_list.rename(columns={'ACCOUNT_ID': 'account'}, inplace=True)

    # For every row in noise_list, set the 'is_sar' value in data to 1 if 0 and 0 if 1
    for index, row in noise_list.iterrows():
        data.loc[data['account'] == row['account'], 'is_sar'] = 1 - data.loc[data['account'] == row['account'], 'is_sar']

    # Save the new data to a csv file
    return data

def generate_noised_dataset(DATASET):
    node_path = f'../gnn/data/{DATASET}/nodes.csv'
    noise_path = f'../AMLsim/outputs/{DATASET}/noise_log.csv'

    node_log = pandas.read_csv(node_path)
    noise_log = pandas.read_csv(noise_path)

    noised_node_log = apply_noise(node_log,noise_log)

    noised_node_log.to_csv(f'../gnn/data/{DATASET}/noised_nodes.csv', index=False)


def main():
    generate_noised_dataset('100K_accts_MID5')


if __name__ == '__main__':
    main()