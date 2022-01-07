from Embeddings.Node2Vec import node_representations
from Dataset import gsp_nilm_dataset
import torch_geometric
from Gnn_Models.model import GCN
from Gnn_Models import model
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit


def main():
    dataset = gsp_nilm_dataset.NilmDataset(root='data', filename='dishwasher.csv', window=20, sigma=20)
    data = dataset[0]
    print(data)
    degrees = torch_geometric.utils.degree(data.edge_index[0])
    #     n_cuts = torch_geometric.utils.normalized_cut(edge_index=data.edge_index, edge_attr=data.edge_attr)
    #     data.x = degrees
    #     print(data)
    data.x = degrees
    print(data.x)
    nodes = node_representations(data)
    # data.x = nodes

    # transform = RandomLinkSplit(is_undirected=True)
    # train_data, val_data, test_data = transform(data)
    # # train_data, val_data, test_data = transform(data)
    # print(train_data, val_data, test_data)

    model = GCN(in_channels=2, hidden_channels=2,
                out_channels=len(np.unique(data.y)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    epochs = 20

    for epoch in range(1, epochs):
        model.train()

    train_acc, test_acc = model.test()

    print('#' * 70)
    print('Train Accuracy: %s' % train_acc)
    print('Test Accuracy: %s' % test_acc)
    print('#' * 70)
    exit('------------------------TEST--------------------------')

    for epoch in range(1, 10):
        loss = model.train(model, optimizer, train_data, criterion)
        acc = model.test(model, optimizer, train_data)
        # train_losses.append(loss)
        # val_losses.append(acc)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    print(model)


if __name__ == '__main__':
    main()
