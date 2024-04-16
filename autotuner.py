import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from torch_geometric.nn import global_add_pool, GATConv, CGConv, GCNConv, Sequential
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix
import pickle
from collections import Counter
from itertools import product

# if torch.backends.mps.is_built():
#     device = torch.device("mps")
# else:
device = "cpu"

def move_to_device(data, device):
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    return data

####################### LOADING DATA #######################

with open('data/bio_data.pkl', 'rb') as file:
    bio_data = pickle.load(file)

# 0 is non-biodegradable, 1 is biodegradable
with open('data/bio_labels.pkl', 'rb') as file:
    bio_labels = pickle.load(file)

X_train, X_test, y_train, y_test = train_test_split(bio_data, bio_labels, test_size=0.2, random_state=49)

####################### CLASS BALANCING - OVERSAMPLING #######################

# 0 is non-biodegradable, 1 is biodegradable
class_counts = Counter(y_train)

class_data = {}
for data, label in zip(X_train, y_train):
    if label not in class_data:
        class_data[label] = []
    class_data[label].append(data)

minority_class = min(class_counts, key=class_counts.get)
majority_class = max(class_counts, key=class_counts.get)

oversampled_data = class_data[minority_class].copy()
num_samples_to_add = class_counts[majority_class] - class_counts[minority_class]
oversampled_data += class_data[minority_class][:num_samples_to_add]
balanced_data = oversampled_data + class_data[majority_class]
balanced_labels = [minority_class] * len(oversampled_data) + [majority_class] * len(class_data[majority_class])

combined_data = list(zip(balanced_data, balanced_labels))
random.shuffle(combined_data)
balanced_data, balanced_labels = zip(*combined_data)

balanced_X_train = list(balanced_data)
balanced_y_train = list(balanced_labels)

class_counts = Counter(balanced_y_train)

####################### MODEL DECLARATION #######################

class GraphEncoder(nn.Module):
    def __init__(self, gcn_layer_configs, dropout):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        layers = []
        input_dim = 10
        for out_channels in gcn_layer_configs:
            layers.append(GATConv(input_dim, out_channels).to(device))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.Dropout(p=self.dropout))
            input_dim = out_channels  # Update input_dim for the next layer
        self.layers = layers
        # self.network = nn.Sequential(*layers)
        
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            if type(layer) == GATConv:
                x = layer(x, edge_index, edge_attr).to(device)
            else:
                x = layer(x)
            # x = nn.ReLU()(x)
            # x = nn.BatchNorm1d(layer.out_channels)(x)
            # x = nn.Dropout(p=self.dropout)(x)
        x = x.to(device)
        return x
    
class BioClassifier(nn.Module):
    def __init__(self, lin_layer_configs, dropout, encoder):
        super(BioClassifier, self).__init__()
        self.dropout = dropout
        self.encoder = encoder
        layers = []
        input_dim = encoder.layers[-4].out_channels # output dim of the last GATConv layer
        for out_features in lin_layer_configs:
            layers.append(torch.nn.Linear(input_dim, out_features))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=self.dropout))
            input_dim = out_features
        layers.append(torch.nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        x = x.float()
        x = self.encoder(x, edge_index, edge_attr)
        x = global_add_pool(x, data.batch)
        x = x.to(device)
        x = self.network(x)
        x = torch.sigmoid(x)
        return x.squeeze(dim=1)
    
def train(model, train_loader, optimizer, epochs, criterion):
    train_loss_list = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            graph, bio = batch[0], batch[1].to(device)
            graph = move_to_device(graph, device)
            output = model(graph).to(device)
            loss = criterion(output.to(torch.float), bio.to(torch.float))
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
    train_loss = sum(train_loss_list) / len(train_loss_list)
    return train_loss

def evaluate(model, test_loader, criterion):
    model.eval()
    val_loss_list = []
    with torch.no_grad():
        for batch in test_loader:
            graph, bio = batch[0], batch[1].to(device)
            graph = move_to_device(graph, device)
            output = model(graph).to(device)
            loss = criterion(output.to(torch.float), bio.to(torch.float))
            val_loss_list.append(loss.item())
    val_loss = sum(val_loss_list) / len(val_loss_list)
    return val_loss

def get_metrics(model, test_dataset):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        correct = 0
        total = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        for batch in test_dataset:
            graph, label = batch[0], batch[1]
            graph = move_to_device(graph, device)
            output = model(graph)
            predicted = torch.round(output)
            total += 1
            y_true.append(int(label))
            y_pred.append(predicted)
            if predicted == int(label):
                correct += 1
            if predicted == 1:
                if int(label) == 1:
                    TP += 1
                else:
                    FP += 1
            if predicted == 0:
                if int(label) == 1:
                    FN += 1
                else:
                    TN += 1
    accuracy = correct / total
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    Ba = (Sn + Sp) / 2
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return accuracy, Ba, roc_auc, precision

####################### RUNNING EXPERIMENTS #######################

train_dataset = list(zip(balanced_X_train, balanced_y_train))
test_dataset = list(zip(X_test, y_test))

# Define the hyperparameter grid for layer configurations
gat_layers = [
    [512], [512, 512, 512, 512], [512, 512, 512], [512, 256, 128, 64], [512, 256, 128], [256, 256, 256], [256, 256], [256, 128], [256], [512]
]
lin_layers = [
    [512, 512, 512], [512, 512], [512, 256], [256, 128], [128, 64], [512], [256], [128], [64]
]
other_params_grid = {
    'epochs': [50, 100, 150],
    'learning_rate': [0.0001, 0.00001, 0.000001, 0.0000001],
    'batch_size': [8, 16, 32, 64],
    'dropout': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# List to store results
results = []

# Iterate over all combinations of parameters
for gat_layer_config, lin_layer_config in product(gat_layers, lin_layers):
    for values in product(*other_params_grid.values()):
        param_dict = dict(zip(other_params_grid.keys(), values))
        encoder = GraphEncoder(gat_layer_config, param_dict['dropout']).to(device)
        model = BioClassifier(lin_layer_config, param_dict['dropout'], encoder).to(device)
        train_loader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=param_dict['batch_size'], shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
        train(model, train_loader, optimizer, param_dict['epochs'], criterion)
        loss = evaluate(model, test_loader, criterion)
        accuracy, Ba, roc_auc, precision = get_metrics(model, test_dataset)
        experiment_result = {**param_dict, 'gcn_layer_config': gat_layer_config, 'lin_layer_config': lin_layer_config, 
                    'loss': loss, 'accuracy': accuracy, 'balanced_accuracy': Ba, 'roc_auc': roc_auc, 'precision': precision}
        results.append(experiment_result)
        print(f'Experiment done! Results: {experiment_result}')

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_tuning_results.csv', index=False)

print("Tuning complete. Results saved to 'model_tuning_results.csv'.")