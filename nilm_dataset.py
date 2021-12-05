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

# from IPython.display import Javascript  # Restrict height of output cell.

# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

# model = GCN(in_channels = dataset.x.shape[1], hidden_channels=dataset.x.shape[1])



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
        self.G = nx.read_graphml(self.raw_paths[0])
        print(len(self.G.nodes), len(self.G.edges))
        # TODO: read graphs below
        # Get node features
        node_feats = self._get_node_features(self.G)
        # Get edge features
        edge_feats = self._get_edge_features(self.G)
        # Get adjacency info
        edge_index = self._get_adjacency_info(self.G)
        # Get labels info
        labels = self._get_labels(nx.get_node_attributes(self.G,
                                                         'state'))  # pass label here. E.g. if it is a column for this graph it could be graph_csv['label']

        # Create data object
        self.data = Data(x=node_feats, edge_index=edge_index, y=labels)
        # self.data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=labels)

        # self.data.num_classes = 2

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

    def _get_edge_features(self, graph):
        """
          This will return a matirx with the gaussian filter kernel of all
          edges
        """

        all_edge_feats = []
        for e in graph.edges(data=True):
            all_edge_feats += [[e[2]['gaussian_kernel']], [e[2]['gaussian_kernel']]]

        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, graph):
        """
        We could also use torch_geometric.from_networkx to create a Data object
        with both adjacency and features, but instead we do it manually here
        """
        nodes = {n: i for i, n in enumerate(graph.nodes())}

        edge_indices = []
        for edge in graph.edges:
            i = nodes[edge[0]]  # get source
            j = nodes[edge[1]]  # get destination
            edge_indices += [[i, j], [j, i]]  # undirected graph

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

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



data = NilmDataset(root='data', filename='dishwaser_20.graphml')
print(data.data.y)
# transform = RandomNodeSplit()
# dataset = transform(data.data)
# print(dataset)

transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(data.data)
# train_data, val_data, test_data = transform(data)
print(train_data, val_data, test_data)

model = GCN(in_channels=train_data.x.shape[1], hidden_channels=train_data.x.shape[1],
            out_channels=len(np.unique(train_data.y)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()



for epoch in range(1, 101):
    loss = train(model)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')



