
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import data
import modules
from data import AmlsimDataset
from modules import GraphSAGE, GCN, GAT  # Ensure GCN is correctly imported here
import torch.optim as optim
import pandas as pd
import os


# Reload your modules if they have been updated
reload(data)
reload(modules)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


dataset_names = ["100K_accts_EASY25_50", "100K_accts_MID5_50", "100K_accts_HARD1_50"]  # List of dataset names

ap_scores = []

def preprocess_data(file_path):
    """Load the CSV file, fill missing values with 0, and return a DataFrame."""
    df = pd.read_csv(file_path)
    if 'bank' in df.columns:
        df = df.drop(columns=['bank'])  # Drop the 'bank' column if it exists
    if 'true_label' in df.columns:
        df = df.drop(columns=['true_label'])  # Drop the 'bank' column if it exists
    df.fillna(0, inplace=True)  # Fill missing values with 0
    return df

def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# Load and normalize data
def load_and_normalize_data(dataset_name):
    base_path = '/data/AML/flib/AMLsim/data/'
    train_node_file = f'{base_path}{dataset_name}/bank/train/nodes.csv'
    train_edge_file = f'{base_path}{dataset_name}/bank/train/edges.csv'
    test_node_file = f'{base_path}{dataset_name}/bank/test/nodes.csv'
    test_edge_file = f'{base_path}{dataset_name}/bank/test/edges.csv'

    # Preprocess nodes and edges
    train_nodes = preprocess_data(train_node_file)
    train_edges = preprocess_data(train_edge_file)
    test_nodes = preprocess_data(test_node_file)
    test_edges = preprocess_data(test_edge_file)

    # Save preprocessed data back to CSV files
    save_preprocessed_data(train_nodes, train_node_file)
    save_preprocessed_data(train_edges, train_edge_file)
    save_preprocessed_data(test_nodes, test_node_file)
    save_preprocessed_data(test_edges, test_edge_file)

    traindata = AmlsimDataset(node_file=train_node_file, edge_file=train_edge_file, node_features=True, node_labels=True).get_data()
    testdata = AmlsimDataset(node_file=test_node_file, edge_file=test_edge_file, node_features=True, node_labels=True).get_data()
    traindata, testdata = traindata.to(device), testdata.to(device)

    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std

    return traindata, testdata



def train_and_evaluate(config, traindata, testdata, dataset_name):
    # Adjustment to accommodate 'n_layers' for GCN
    if config.get('num_heads') is not None:  # This is specific to GAT
        model = config["model_class"](config["in_channels"], config["hidden_channels"], config["out_channels"], config["num_heads"], config["dropout"]).to(device)
    elif config.get('n_layers') is not None:
        model = config["model_class"](config["input_dim"], config["hidden_dim"], config["output_dim"], config["n_layers"], config["dropout"]).to(device)
    else:
        model = config["model_class"](config["input_dim"], config["hidden_dim"], config["output_dim"], config["dropout"]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(traindata)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:  # Evaluation
            model.eval()
            with torch.no_grad():
                out = model(testdata)
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f"Epoch: {epoch+1}, Precision: {precision}, Recall: {recall}")

    # Final evaluation and plot
    AP = evaluate_and_plot(model, testdata, config["plot_name"], dataset_name)

    return AP

def evaluate_and_plot(model, testdata, plot_name, dataset_name):
    model.eval()
    with torch.no_grad():
        out = model(testdata)
        y_scores = F.softmax(out, dim=1).cpu().numpy()[:, 1]  # Probabilities for the positive class
        y_true = testdata.y.cpu().numpy()  # True labels

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall (Positive label: 1)')
    plt.ylabel('Precision (Positive label: 1)')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    # Check if the directory exists, create it if not
    results_dir = "GNN_Results_50"
    os.makedirs(results_dir, exist_ok=True)
    
    # Adjust the file path to include the 'GNN_Results' directory
    plot_path = os.path.join(results_dir, f'{plot_name}_{dataset_name}.png')
    
    # Save the plot
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved as {plot_path}")
    return average_precision




for dataset_name in dataset_names:
    print(f"Processing dataset: {dataset_name}")
    traindata, testdata = load_and_normalize_data(dataset_name)
    #model configurations
    model_configs = {
        "GraphSAGE": {
            "model_class": GraphSAGE,
            "epochs": 200,
            "learning_rate": 0.01,
            "input_dim": traindata.num_node_features,
            "hidden_dim": 128,
            "output_dim": 2,
            "dropout": 0.2,
            "plot_name": "GraphSAGE_precision_recall",
            "n_layers": None, # Specific to GCN
            "num_heads": None,  # Specific to GAT
        },
        "GCN": {
            "model_class": GCN,
            "epochs": 1000,  # Note the difference in the number of epochs
            "learning_rate": 0.05253036287687629,
            "input_dim": traindata.num_node_features,
            "hidden_dim": 128,
            "output_dim": 2,
            "dropout": 0.12858713461385163,
            "plot_name": "GCN_precision_recall",
            "n_layers": 4,  # Specific to GCN
            "num_heads": None,  # Specific to GAT
        },

        "GAT": {
            "model_class": GAT,  
            "epochs": 500,  
            "learning_rate": 0.02183081977444788,
            "in_channels": traindata.num_node_features,
            "hidden_channels": 96,  
            "out_channels": 2,  
            "dropout": 0.16732248390113405,
            "plot_name": "GAT_precision_recall",
            "n_layers": None,  # Specific to GCN
            "num_heads": 4,  # Specific to GAT
        }
    }

    for model_name, config in model_configs.items():
        print(f"Training and evaluating {model_name} on {dataset_name}")
        average_precision = train_and_evaluate(config, traindata, testdata, dataset_name)
        # Append the AP score along with model and dataset name to the list
        ap_scores.append({'Model': model_name, 'Dataset': dataset_name, 'Average Precision': average_precision})

# Convert the list of AP scores to a DataFrame and save to CSV
ap_scores_df = pd.DataFrame(ap_scores)
results_csv_path = os.path.join("GNN_Results_50", "ap_scores_summary.csv")
ap_scores_df.to_csv(results_csv_path, index=False)
print(f"AP scores summary saved to {results_csv_path}")

