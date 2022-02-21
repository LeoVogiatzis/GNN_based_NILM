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
from Embeddings.Auto_Encoder import pairwise_auto_encoder
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    print(out)
    print(train_data.y)
    # loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    loss = criterion(out, train_data.y.view(-1, 1))
    loss.backward(retain_graph=True)  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(model):
    model.eval()
    out = model(test_data.x, test_data.edge_index)
    test_loss = criterion(out, test_data.y.view(-1, 1))
    # Derive ratio of correct predictions.
    return test_loss


dataset = gsp_nilm_dataset.NilmDataset(root='data', filename='mains_2.csv', window=20, sigma=20)
data = dataset[0]
print(data)

embedding_method = ''
if embedding_method == 'Node2Vec':
    embeddings = node_representations(data)
    data.x = embeddings.data

elif embedding_method == 'AE':
    data = pairwise_auto_encoder(data)

else:
    print(data.x)

data.y = data.y.type(torch.FloatTensor)
print(data.x)

transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(data)
print(train_data, val_data, test_data)

model = GCN(in_channels=4, hidden_channels=4, out_channels=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()
epochs = 20
train_losses = []
val_losses = []
for epoch in range(1, 100):
    loss = train(model)
    # acc = test(model, test_data, criterion)
    test_loss = test(model)
    train_losses.append(loss.item())
    val_losses.append(test_loss.item())
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
print(model)
results = model(data.x, data.edge_index)
print(results)
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(val_losses, label="val")
plt.plot(train_losses, label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
print('End Pipeline')
