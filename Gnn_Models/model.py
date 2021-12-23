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
    # loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    loss = criterion(out, train_data.y)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(model, optimizer, dataset):
    model.eval()
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

# def train(model, optimizer, data, criterion):
#     model.train()
#     optimizer.zero_grad()
#     F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
#     optimizer.step()
#
#
# @torch.no_grad()
# def test(model, data):
#     model.eval()
#     logits = model()
#     mask1 = data['train_mask']
#     pred1 = logits[mask1].max(1)[1]
#     acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
#     mask = data['test_mask']
#     pred = logits[mask].max(1)[1]
#     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#     return acc1, acc
