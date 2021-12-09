#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from torch_geometric.utils import from_scipy_sparse_matrix

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


def graph_creation(data_vec, sigma,
                   window=1000):  # this defines  number of observations in a window, the algorthm works in a sliding window manner
    data_aggr = []

    for k in range(0, int(np.floor(len(data_vec) / window))):
        data_aggr.append(np.mean(data_vec[k * window:((k + 1) * window)]))

    if (len(data_vec) % window > 0):
        data_aggr.append(np.mean(data_vec[int(np.floor(len(data_vec) / window)) * window:]))
    delta_p = [np.round(data_aggr[i + 1] - data_aggr[i], 2) for i in range(0, len(data_aggr) - 1)]
    Am = np.zeros((len(delta_p), len(delta_p)))
    for i in range(0, Am.shape[0]):
        for j in range(0, Am.shape[1]):
            Am[i, j] = math.exp(-((delta_p[i] - delta_p[j]) / sigma) ** 2)
    return Am, delta_p


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)

        return x


def train(model):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    # out = model(dataset.x, dataset.edge_index)  # Perform a single forward pass.
    out = model(train_data.x, train_data.edge_index)

    # loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    loss = criterion(out, train_data.y)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(model):
    model.eval()
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def main():
    df = pd.read_csv('data/house5.csv')  # read demo file with aggregated active power

    # Please read the paper to undertand following parameters. Note initial values of these parameters depends on the appliances used and the frequency of usage.
    sigma = 20;
    ri = 0.15
    T_Positive = 20;
    T_Negative = -20;
    # Following parameters alpha and beta are used in Equation 15 of the paper
    # alpha define weight given to magnitude and beta define weight given to time
    alpha = 0.5
    beta = 0.5
    # this defines the  minimum number of times an appliance is set ON in considered time duration
    instancelimit = 3

    # %%
    main_val = df['dishwaser_20'].values  # get only readings
    # main_ind = df.index  # get only timestamp
    data_vec = main_val
    threshold = 2000  # threshold of DTW algorithm used for appliance power signature matching

    Am, delta_p = graph_creation(data_vec, sigma, window=77)
    edge_indices = torch.tensor(Am)
    drift = torch.tensor(delta_p, dtype=torch.float)
    labels = [0 if i == 0.0 else 1 for i in drift]
    # labels = [1 if (i > 0 and i != 0) else -1 for i in drift if i!=0]
    labels = torch.tensor(labels, dtype=torch.int64)
    # edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    dataset = Data(x=drift, edge_index=edge_indices, y=labels)
    train_mask = Data(x=dataset.x[0:300],
                      edge_index=edge_indices,
                      num_classes=2,
                      y=labels
                      )
    test_mask = Data(x=dataset.x[300:],
                     edge_index=edge_indices,
                     num_classes=2,
                     y=labels
                     )
    # transform = RandomLinkSplit(is_undirected=True)
    # train_data, val_data, test_data = transform(dataset)
    # train_data, val_data, test_data = transform(data)
    print(train_mask, test_mask)

    model = GCN(in_channels=train_mask, hidden_channels=len(train_mask),
                out_channels=len(np.unique(train_mask.y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


if __name__ == '__main__':
    main()
