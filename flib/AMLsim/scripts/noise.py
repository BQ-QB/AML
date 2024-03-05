import csv
import random
import os
""""
# Define the path to the CSV file
csv_file_path = "/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/tx_log.csv"
count = 0
# Open the CSV file
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)
    
    # Read the header to identify the column index for 'isSAR'
    header = next(csv_reader)
    isSAR_index = header.index('isSAR')
    nameOrig_index = header.index('nameOrig')
    
    # Iterate over each row in the CSV
    for row in csv_reader:
        # Check if 'isSAR' equals '0'
        if row[isSAR_index] == '1' and row[nameOrig_index]=='5370':
            # Print the row
            count = count + 1  
            print(row)
            print(count)
"""
def flip_sar_values_accounts_csv(false_to_true_percentage=5, true_to_false_percentage=5):
    
    modified_data = []
    header = None
    csv_input_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/accounts.csv'
    csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/noise_accounts.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    with open(csv_input_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Assuming the first row is the header
        is_sar_index = header.index('IS_SAR')  # Identify the 'is_sar' column index

        for row in reader:
            is_sar_value = row[is_sar_index].lower() in ('true', '1')
            flip_chance = random.randint(1, 100)  # Generate a random percentage chance
            
            # Decide whether to flip based on the current value and specified percentages
            if is_sar_value and flip_chance <= true_to_false_percentage:
                row[is_sar_index] = 'false'
            elif not is_sar_value and flip_chance <= false_to_true_percentage:
                row[is_sar_index] = 'true'
                
            modified_data.append(row)

    # Write the modified data to a new CSV file
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:  # Write the header first if it exists
            writer.writerow(header)
        writer.writerows(modified_data)

def flip_sar_values_by_reason():

    modified_data = []
    header = None
    csv_input_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/alert_members.csv'
    csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/noise_accounts.csv'

    flip_percentages = {
    'fan_out': 10,
    'fan_in': 20,
    'cycle': 30,
    'bipartide': 10,
    'stack': 20,
    'scatter_gather': 30,
    'gather_scatter': 10,
    # Add the remaining 6 reasons and their flip percentages here
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    with open(csv_input_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Assuming the first row is the header
        is_sar_index = header.index('is_sar')  # Identify the 'is_sar' column index
        reason_index = header.index('reason')  # Identify the 'reason' column index

        for row in reader:
            reason = row[reason_index]
            if reason in flip_percentages:
                flip_chance = random.randint(1, 100)  # Generate a random percentage chance
                # Flip 'is_sar' value if the random chance is within the specified percentage for the reason
                if flip_chance <= flip_percentages[reason]:
                    current_value = row[is_sar_index].lower() in ('true', '1')
                    row[is_sar_index] = 'False' if current_value else 'True'
            modified_data.append(row)

    # Write the modified data to a new CSV file
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:  # Write the header first if it exists
            writer.writerow(header)
        writer.writerows(modified_data)

def flip_sar_based_on_reason_and_percentage():
    reasons_csv_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/alert_members.csv'
    main_csv_input_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/accounts.csv'
    csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/noise_accounts.csv'

    flip_percentages = {
        'fan_out': 50,
        'fan_in': 50,
        'cycle': 50,
        'bipartide': 50,
        'stack': 50,
        'scatter_gather': 50,
        'gather_scatter': 50,
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    # Read reasons and collect account IDs per reason
    reason_account_ids = {reason: [] for reason in flip_percentages}
    with open(reasons_csv_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reason = row['reason']
            accountID = row['accountID']  # Ensure this matches the column name in your CSV
            if reason in reason_account_ids:
                reason_account_ids[reason].append(accountID)
    
    # Decide which accounts to flip based on the specified percentages
    accounts_to_flip = set()
    for reason, account_ids in reason_account_ids.items():
        flip_count = int(len(account_ids) * (flip_percentages[reason] / 100))
        accounts_to_flip.update(random.sample(account_ids, flip_count))

    # Read the main CSV, flip 'is_sar' where accountID matches
    print(accounts_to_flip)
    modified_data = []
    with open(main_csv_input_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        for row in reader:
            if row['ACCOUNT_ID'] in accounts_to_flip:
                current_value = row['IS_SAR'].lower() in ('true', '1')
                row['IS_SAR'] = 'false' if current_value else 'true'
            modified_data.append(row)
       

    # Write the modified data to a new CSV file, ensuring fieldnames match exactly
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(modified_data)

def delete_accounts_based_on_reason_and_percentage():
    reasons_csv_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/alert_members.csv'
    main_csv_input_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/accounts.csv'
    transactions_csv_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/transactions.csv'
    reasons_csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/filtered_alert_members.csv'
    accounts_csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/filtered_accounts.csv'
    transactions_csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/filtered_transactions.csv'

    delete_percentages = {
        'fan_out': 100,
        'fan_in': 100,
        'cycle': 100,
        'bipartite': 100,
        'stack': 100,
        'scatter_gather': 100,
        'gather_scatter': 100,
    }

    # Ensure the output directories exist
    os.makedirs(os.path.dirname(reasons_csv_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(accounts_csv_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(transactions_csv_output_path), exist_ok=True)

    # Read reasons and collect account IDs per reason
    reason_account_ids = {reason: [] for reason in delete_percentages}
    with open(reasons_csv_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reason = row['reason']
            accountID = row['accountID']  # Ensure this matches the column name in your CSV
            if reason in reason_account_ids:
                reason_account_ids[reason].append(accountID)

    # Decide which accounts to delete based on the specified percentages
    accounts_to_delete = set()
    for reason, account_ids in reason_account_ids.items():
        delete_count = int(len(account_ids) * (delete_percentages[reason] / 100))
        accounts_to_delete.update(random.sample(account_ids, delete_count))

    # Filter and write the modified alarm (reasons) data to a new CSV file
    with open(reasons_csv_path, mode='r', newline='') as csvfile, open(reasons_csv_output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['accountID'] not in accounts_to_delete:
                writer.writerow(row)

    # Filter and write the modified accounts data to a new CSV file
    with open(main_csv_input_path, mode='r', newline='') as csvfile, open(accounts_csv_output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['ACCOUNT_ID'] not in accounts_to_delete:
                writer.writerow(row)

    # Now, filter and write the modified transactions data to a new CSV file
    with open(transactions_csv_path, mode='r', newline='') as csvfile, open(transactions_csv_output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['src'] not in accounts_to_delete and row['dst'] not in accounts_to_delete:
                writer.writerow(row)

def delete_transactions_reasons():
    reasons_csv_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/alert_members.csv'
    transactions_csv_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/tmp/10K_accts/transactions.csv'
    transactions_csv_output_path = '/home/edgelab/UnreliableLabels/flib/AMLsim/outputs/10K_accts/filtered_transactions.csv'

    delete_percentages = {
        'fan_out': 100,
        'fan_in': 100,
        'cycle': 100,
        'bipartite': 100,
        'stack': 100,
        'scatter_gather': 100,
        'gather_scatter': 100,
    }

    # Ensure the output directories exist
    os.makedirs(os.path.dirname(transactions_csv_output_path), exist_ok=True)

    # Read reasons and collect account IDs per reason
    reason_account_ids = {reason: [] for reason in delete_percentages}
    with open(reasons_csv_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reason = row['reason']
            accountID = row['accountID']  # Ensure this matches the column name in your CSV
            if reason in reason_account_ids:
                reason_account_ids[reason].append(accountID)

    # Decide which accounts to delete based on the specified percentages
    accounts_to_delete = set()
    for reason, account_ids in reason_account_ids.items():
        delete_count = int(len(account_ids) * (delete_percentages[reason] / 100))
        accounts_to_delete.update(random.sample(account_ids, delete_count))

    # Now, filter and write the modified transactions data to a new CSV file
    with open(transactions_csv_path, mode='r', newline='') as csvfile, open(transactions_csv_output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['src'] not in accounts_to_delete and row['dst'] not in accounts_to_delete:
                writer.writerow(row)
    



#flip_sar_values_accounts_csv() #Behåll denna
#flip_sar_values_by_reason() #EV skrota denna
#flip_sar_based_on_reason_and_percentage() #Behåll denna
#delete_accounts_based_on_reason_and_percentage()
#delete_transactions_reasons()

#TODO fixa tids noise, tror detta kräver att vi bestämmer oss om data formatering innan GNN
#TODO fixa missing labels 

