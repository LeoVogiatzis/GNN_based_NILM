import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from Embeddings.Node2Vec import node_representations
from Dataset import gsp_dataset
import seaborn as sns
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import matplotlib.pyplot as plt


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,
                            out_channels=32,
                            periods=periods,
                            batch_size=batch_size)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


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


# dataset = gsp_dataset.NilmDataset(root='data', filename='kitchen_outlets_house_2_x.csv', window=10, sigma=20)
data = torch.load('/home/leonidas/PycharmProjects/GNN_based_NILM/data/processed/kitchen_outlets_house_2_x.pt')
print(data)

transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(data)
print(train_data, val_data, test_data)
model = TemporalGNN(node_features=4, periods=12, batch_size=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()
epochs = 20
train_losses = []
val_losses = []
for epoch in range(1, 200):
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
exit()
from torch_geometric_temporal.signal import temporal_signal_split

# transform = RandomLinkSplit(is_undirected=True)
# train_dataset, val_data, test_dataset = transform(data)
# print(train_dataset, val_data, test_dataset)

train_dataset, test_dataset = temporal_signal_split(data, train_ratio=0.8)
# train_dataset, test_dataset = temporal_signal_split(data, train_ratio=0.8)

print("Number of train buckets: ", len(set(train_dataset)))

# GPU support
# device = torch.device('cpu') # cuda
subset = 2000

# Create model and optimizers
model = TemporalGNN(node_features=4, periods=12, batch_size=2)  # .to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

print("Running training...")
for epoch in range(10):
    loss = 0
    step = 0
    for snapshot in train_dataset:
        snapshot = snapshot  # .to(device)
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
        step += 1
        if step > subset:
            break

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

model.eval()
loss = 0
step = 0
horizon = 288

# Store for analysis
predictions = []
labels = []

for snapshot in test_dataset:
    snapshot = snapshot  # .to(device)
    # Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index)
    # Mean squared error
    loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
    # Store for analysis below
    labels.append(snapshot.y)
    predictions.append(y_hat)
    step += 1
    if step > horizon:
        break

loss = loss / (step + 1)
loss = loss.item()
print("Test MSE: {:.4f}".format(loss))

import numpy as np

sensor = 123
timestep = 11
preds = np.asarray([pred[sensor][timestep].detach().cpu().numpy() for pred in predictions])
labs = np.asarray([label[sensor][timestep].cpu().numpy() for label in labels])
print("Data points:,", preds.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
sns.lineplot(data=preds, label="pred")
sns.lineplot(data=labs, label="true")
plt.show()
