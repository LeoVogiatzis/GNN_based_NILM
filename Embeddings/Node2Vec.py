import networkx as nx
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
import torch


from torch_geometric.nn import Node2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(loader, model, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        # optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc


def node_representations(data):
    # data = dataset[0]
    # from torch_geometric.datasets import Planetoid
    # from torch_geometric.transforms import NormalizeFeatures
    # dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    # data = dataset[0]
    # print(data.y)
    print(data.edge_index)
    print(data.train_mask)
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    for idx, (pos_rw, neg_rw) in enumerate(loader):
        print(idx, pos_rw.shape, neg_rw.shape)
    idx, (pos_rw, neg_rw) = next(enumerate(loader))
    train_losses = []
    val_losses = []
    for epoch in range(1, 10):
        loss = train(loader, model, optimizer)
        acc = test(model, data)
        train_losses.append(loss)
        val_losses.append(acc)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    print(model)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model
