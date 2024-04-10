import pandas
import os

def apply_noise(data,noise_list):
    # For data add column 'true_label' as a copy of 'is_sar'
    data['true_label'] = data['is_sar']
    noise_list.rename(columns={'ACCOUNT_ID': 'account'}, inplace=True)

    counter1 = 0
    counter2 = 0
    # For every row in noise_list, set the 'is_sar' column in data to 0 if already 1 and 1 if already 0
    for index, row in noise_list.iterrows():
        if data.loc[data['account'] == row['account'], 'is_sar'].values[0] == 1:
            data.loc[data['account'] == row['account'], 'is_sar'] = 0
            counter1 += 1
        else:
            data.loc[data['account'] == row['account'], 'is_sar'] = 1
            counter2 += 1

    # Save the new data to a csv file
    return data, counter1,counter2

def generate_noised_dataset(DATASET,RATIO):

    node_path = f'../gnn/data_statmodels/{DATASET}/nodes.csv'
    noise_path = f'../noise/{DATASET}/{RATIO}/'

    noises = os.listdir(noise_path)
    
    all_noise = pandas.DataFrame()

    node_log = pandas.read_csv(node_path)

    # Create the directory if it does not exist
    if not os.path.exists(f'../datasets_statmodels/{DATASET}/no_noise'):
        os.makedirs(f'../datasets_statmodels/{DATASET}/no_noise')
    node_log.to_csv(f'../datasets_statmodels/{DATASET}/no_noise/nodes.csv', index=False)

    for noise in noises:
        noise_log = pandas.read_csv(noise_path + noise)

        noised_node_log,counter1,counter2= apply_noise(node_log,noise_log)

        # Combine all the noise dataframes into one dataframe
        all_noise = pandas.concat([all_noise, noise_log])

        # Create the directory if it does not exist
        if not os.path.exists(f'../datasets_statmodels/{DATASET}/{RATIO}'):
            os.makedirs(f'../datasets_statmodels/{DATASET}/{RATIO}')
        noised_node_log.to_csv(f'../datasets_statmodels/{DATASET}/{RATIO}/nodes_{noise[6:-4]}_noised.csv', index=False)

        print(f"For dataset {DATASET} with ratio {RATIO} and noise {noise[6:-4]}: {counter1} 1s were flipped to 0s and {counter2} 0s were flipped to 1s")

    full_noise,counter1,counter2 = apply_noise(node_log,all_noise)

    print(f"For dataset {DATASET} with ratio {RATIO} and noise FULL: {counter1} 1s were flipped to 0s and {counter2} 0s were flipped to 1s")

    # Create the directory if it does not exist
    if not os.path.exists(f'../datasets_statmodels/{DATASET}/{RATIO}'):
        os.makedirs(f'../datasets_statmodels/{DATASET}/{RATIO}')
    full_noise.to_csv(f'../datasets_statmodels/{DATASET}/{RATIO}/nodes_FULL_noised.csv', index=False)


def main():
    DATASETS = ['100K_accts_MID5','100K_accts_HARD1','100K_accts_EASY25']
    RATIOS = ['0.1','0.25']
    for DATASET in DATASETS:
        for RATIO in RATIOS:
            generate_noised_dataset(DATASET,RATIO)


if __name__ == '__main__':
    main()