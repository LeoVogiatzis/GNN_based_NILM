import glob

import matplotlib.pyplot as plt
# from Gnn_Models.model import GCN
# from Gnn_Models import model
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

from Dataset import gsp_dataset
from Embeddings.Auto_Encoder import pairwise_auto_encoder
from Embeddings.Node2Vec import node_representations


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


def ground_truth(main_val, data):
    data_aggr = []
    window = 20
    for k in range(0, int(np.floor(len(main_val) / window))):
        data_aggr.append(np.mean(main_val[k * window:((k + 1) * window)]))

    if (len(main_val) % window > 0):
        data_aggr.append(np.mean(main_val[int(np.floor(len(main_val) / window)) * window:]))
    delta_p = [np.round(data_aggr[i + 1] - data_aggr[i], 2) for i in range(0, len(data_aggr) - 1)]
    # freq_data = data.y.detach().numpy() / np.linalg.norm(data.y.detach().numpy())
    freq_data = delta_p / np.linalg.norm(delta_p)
    plt.figure(figsize=(10, 5))
    plt.title("Consumption")
    # plt.plot(val_losses, label="val")
    print(data.y.detach().numpy())
    plt.plot(list(data.y.detach().numpy()), label="fourier_transform")
    plt.show()
    plt.plot(freq_data, label="ground_truth")
    plt.xlabel("time")
    plt.ylabel("power_con")
    plt.legend()
    plt.show()

    return data_aggr


def conventional_ml(train_data):
    regr = RandomForestRegressor(n_estimators=10, random_state=0)
    regr.fit(np.array(train_data.x), np.array(train_data.y).ravel())
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(np.array(train_data.y), regr.predict(np.array(train_data.x)).reshape(-1, 1))
    print(mse)
    return regr.predict(np.array(train_data.x)).reshape(-1)


# df = pd.read_csv('/home/leonidas/PycharmProjects/GNN_based_NILM/data/raw/sample_house2.csv',
#                  usecols=['Unnamed: 0', 'kitchen_outlets_house_2_x', 'lighting_house_2', 'refrigerator_house_2',
#                           'microwave_house_2'])
#
# df = df.set_index(pd.DatetimeIndex(df['Unnamed: 0']))
# print(df.head(1))
# print(df.shape[0])
# df = df.resample('1T').sum()
# print(df.shape[0])
# x = 1
#
#
# def write_applince_files(df):
#     for i in df.columns:
#         df[i].to_frame().to_csv('./data/raw/' + str(i) + '.csv', index=False)
#
#
# write_applince_files(df)

# 28/04/11 30/04/11
path = r'/home/leonidas/PycharmProjects/GNN_based_NILM/data/raw'
all_files = glob.glob(path + "/*.csv")
houses = [filename for filename in all_files]
# appliance = pd.read_csv('/home/leonidas/PycharmProjects/GNN_based_NILM/data/raw/lighting_23.csv', index_col=0)
# main_val = appliance['lighting_23'].values

index = 0
for filename in all_files:
    dataset = gsp_dataset.NilmDataset(root='data', filename=f'{filename}', window=20, sigma=20)
    if index > 0:
        index += 1
    data = dataset[index]
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

    # consumption = ground_truth(main_val, data)
    print(data.x)

    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)
    print(train_data, val_data, test_data)

    pred = conventional_ml(train_data)

    plt.title("Predicted/ G-truth")
    plt.plot(pred, label="pred")
    plt.plot(data.y.view(-1, 1), label="g_truth", alpha=0.5)
    plt.xlabel("timestep")
    plt.ylabel("delta_p")
    plt.legend()
    plt.show()
    exit()

    from utils import mse

    y_true = data.y.cpu().detach().numpy()
    y_hat = np.mean(y_true)
    print(mse(np.array([y_hat] * y_true.shape[0]), y_true))

    # exit('By Marinos')
    model = GCN(in_channels=4, hidden_channels=4, out_channels=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    epochs = 20
    train_losses = []
    val_losses = []
    print(np.unique(data.y.view(-1, 1)))

    for epoch in range(1, 10):
        loss = train(model)
        # acc = test(model, test_data, criterion)
        test_loss = test(model)
        train_losses.append(loss.item())
        val_losses.append(test_loss.item())
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    print(model)
    results = model(data.x, data.edge_index)
    results = results.detach().numpy().reshape(-1)
    print(results)
    plt.title("Predicted/ G-truth")
    plt.plot(results, label="pred")
    plt.plot(data.y.view(-1, 1), label="g_truth")
    plt.xlabel("timestep")
    plt.ylabel("delta_p")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    print('End Pipeline')
