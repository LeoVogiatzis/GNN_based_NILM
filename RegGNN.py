'''RegGNN regression model architecture.
torch_geometric needs to be installed.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv

from torch.nn import Linear
from torch_geometric.transforms import RandomLinkSplit


class RegGNN(nn.Module):
    '''Regression using a DenseGCNConv layer from pytorch geometric.
       Layers in this model are identical to GCNConv.
    '''

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RegGNN, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gc2 = DenseGCNConv(nhid, 1)
        self.dropout = dropout
        self.LinearLayer = nn.Linear(nclass, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = self.LinearLayer(torch.transpose(x, 2, 1))

        return torch.transpose(x, 2, 1)

    def loss(self, pred, score):
        return F.mse_loss(pred, score)


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # loss = criterion(out[data.train_mask],
    #                  data.y[data.train_mask])
    loss = criterion(out.squeeze(), train_data.y.squeeze())
    # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


data = torch.load('data/processed/kitchen_outlets_house_2_x.pt')
data.num_classes = len(data.y.unique())
print(f'Dataset: {data}:')
print('======================')
print(f'Number of graphs: {len(data)}')
print(f'Number of features: {data.num_features}')
# print(f'Number of classes: {data.num_classes}')

# data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(data)
print(train_data, val_data, test_data)

bat_size = 3
train_ldr = torch.utils.data.DataLoader(train_data,
                                        batch_size=bat_size, shuffle=True)
val_data = torch.utils.data.DataLoader(val_data,
                                       batch_size=bat_size, shuffle=True)
test_data = torch.utils.data.DataLoader(test_data,
                                        batch_size=bat_size, shuffle=True)

model = RegGNN(nfeat=4, nhid=4, nclass=data.num_classes, dropout=0.1)
print(model)

# model = MLP(hidden_channels=16)
criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)  # Define optimizer.
data.y = data.y.type(torch.FloatTensor)

for epoch in range(1, 100):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
