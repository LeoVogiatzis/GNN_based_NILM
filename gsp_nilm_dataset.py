import networkx as nx
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm

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
import math


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


class NilmDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """2
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(NilmDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        # TODO: read graphs below
        self.data = pd.read_csv(self.raw_paths[0])
        window = 20;
        sigma = 20;
        main_val = self.data['dishwaser_20'].values  # get only readings
        data_vec = main_val
        edge_index, drift = self._get_adjacency_info(data_vec, window, sigma)

        edge_indices = torch.tensor(edge_index)
        drift = np.asarray(drift)
        drift = drift.reshape((-1, 1))
        drift = torch.tensor(drift, dtype=torch.float)
        labels = [0 if i == 0.0 else 1 for i in drift]
        # labels = [1 if (i > 0 and i != 0) else -1 for i in drift if i!=0]
        labels = torch.tensor(labels, dtype=torch.int64)
        # edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        self.data = Data(x=drift, edge_index=edge_indices, y=labels,
                    #  train_mask=[2000], test_mask=[2000]
                    )

        if self.test:
            torch.save(self.data, os.path.join(self.processed_dir, 'data_test_0.pt'))
        else:
            torch.save(self.data, os.path.join(self.processed_dir, 'data_0.pt'))

    def _get_node_features(self, graph):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]

        We could also use torch_geometric.from_networkx to create a Data object
        with both adjacency and features, but instead we do it manually here
        """
        all_node_feats = list(nx.get_node_attributes(graph, 'drift').values())

        all_node_feats = np.asarray(all_node_feats)
        all_node_feats = all_node_feats.reshape((-1, 1))
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_adjacency_info(self, data_vec, window, sigma):
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

    def _get_labels(self, labels):
        labels = list(labels.values())
        labels = np.asarray(labels)
        return torch.tensor(labels, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def main():
    data = NilmDataset(root='data', filename='dishwasher.csv')
    print(data.data.y)
    print(data.data)
    x = 1


if __name__ == '__main__':
    main()
