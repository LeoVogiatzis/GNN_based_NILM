from Embeddings.Node2Vec import node_representations
from Dataset import gsp_nilm_dataset
import torch_geometric
from Gnn_Models import model


def main():
    dataset = gsp_nilm_dataset.NilmDataset(root='data', filename='dishwasher.csv', window=20, sigma=20)
    data = dataset[0]
    print(data)
    nodes = node_representations(data)
    data.x = nodes
    model(data)


if __name__ == '__main__':
    main()
