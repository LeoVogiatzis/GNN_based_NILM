from Embeddings.Node2Vec import node_representations
from Dataset import gsp_nilm_dataset
import torch_geometric

# from Gnn_Models.model import GCN
# from Gnn_Models import model

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


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


def train(model, optimizer, train_data, criterion):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    # out = model(dataset.x, dataset.edge_index)  # Perform a single forward pass.
    out = model(train_data.x, train_data.edge_index)
    print(out)
    print(train_data.y)
    # loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    loss = criterion(out, train_data.y.view(-1, 1))
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(model, test_data, criterion):
    model.eval()
    out = model(test_data.x, test_data.edge_index)
    test_loss = criterion(out, test_data.y.view(-1, 1))
    # Derive ratio of correct predictions.
    return test_loss


def main():
    dataset = gsp_nilm_dataset.NilmDataset(root='data', filename='lighting_house_2.csv', window=20, sigma=20)
    data = dataset[0]
    print(data)
    # degrees = torch_geometric.utils.degree(data.edge_index[0])
    #     n_cuts = torch_geometric.utils.normalized_cut(edge_index=data.edge_index, edge_attr=data.edge_attr)
    #     data.x = degrees
    #     print(data)
    # data.x = degrees
    print(data.x)
    # data.x = degrees.reshape((-1, 1))
    embeddings = node_representations(data)
    data.x = embeddings.data

    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)
    print(train_data, val_data, test_data)

    model = GCN(in_channels=2, hidden_channels=2, out_channels=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    epochs = 20

    for epoch in range(1, 10):
        loss = train(model, optimizer, train_data, criterion)
        acc = test(model, test_data, criterion)
        # train_losses.append(loss)
        # val_losses.append(acc)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    print(model)
    results = model(data.x, data.edge_index)
    print(results)
    print('End Pipeline')


if __name__ == '__main__':
    main()
