from Embeddings import Node2Vec, Auto_Encoder
from Dataset import gsp_nilm_dataset
import torch_geometric


def main():
    dataset = gsp_nilm_dataset.NilmDataset(root='data', filename='dishwasher.csv', window=20, sigma=20)
    data = dataset[0]
    print(data)


if __name__ == '__main__':
    main()
