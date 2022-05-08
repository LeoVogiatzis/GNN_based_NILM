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
from sklearn.metrics import mean_absolute_error, mean_squared_error


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCN, self).__init__()
#         torch.manual_seed(42)
#
#         # Initialize the layers
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         # First Message Passing Layer (Transformation)
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#
#         # Second Message Passing Layer
#         x = self.conv2(x, edge_index)
#
#         return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(train_data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 100)
        self.linear1 = torch.nn.Linear(100, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear1(x)
        return x


def train(model):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    # out = model(dataset.x, dataset.edge_index)  # Perform a single forward pass.
    out = model(train_data.x, train_data.edge_index)
    print(out)
    print(train_data.y)
    # loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    loss = criterion(out.squeeze(), train_data.y.squeeze()) # view(-1, 1))
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
    # freq_data = delta_p / np.linalg.norm(delta_p)
    plt.figure(figsize=(10, 5))
    plt.title("Consumption")
    # plt.plot(val_losses, label="val")
    print(data.y.detach().numpy())
    plt.plot(list(data.y.detach().numpy()), label="fourier_transform")
    plt.show()
    plt.plot(delta_p, label="ground_truth")
    plt.xlabel("time")
    plt.ylabel("power_con")
    plt.legend()
    plt.show()

    return data_aggr


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def conventional_ml(train_data, test_data):
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [80, 90, 100, 110],
    #     'max_samples': [0.25, 0.5, 0.75],
    #     'max_features': [2, 3, 4],
    #     'min_samples_leaf': [2, 3, 4, 5],
    #     'min_samples_split': [7, 8, 9, 10, 12],
    #     'n_estimators': [50, 100, 200, 300, 1000]
    # }

    from sklearn.model_selection import GridSearchCV
    import warnings
    base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    base_model.fit(np.array(train_data.x), np.array(train_data.y).ravel())
    base_accuracy = evaluate(base_model, np.array(train_data.x), np.array(train_data.y).ravel())
    regr = RandomForestRegressor(random_state=0)

    CV_regr = GridSearchCV(estimator=regr, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, return_train_score=True)

    with warnings.catch_warnings(record=True) as w:
        try:
            CV_regr.fit(np.array(train_data.x), np.array(train_data.y).ravel())
        except ValueError:
            pass
        # print(repr(w[-1].message))
    # train_data.x.detach().numpy(), train_data.y.detach().numpy().ravel()
    # CV_regr.fit(np.array(train_data.x), np.array(train_data.y).ravel())
    print(f'best parameters: {CV_regr.best_params_}')
    best_grid = CV_regr.best_estimator_

    grid_accuracy = evaluate(best_grid, np.array(train_data.x), np.array(train_data.y).ravel())
    print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

    mse = mean_squared_error(np.array(test_data.y), best_grid.predict(np.array(test_data.x)))  # .reshape(-1, 1)
    mae = mean_absolute_error(np.array(test_data.y), best_grid.predict(np.array(test_data.x)))
    print(f'best_estimator {best_grid}')
    print(f'mse: {mse}')
    print(f'mae: {mae}')
    return best_grid.predict(test_data.x.detach().numpy()).reshape(-1)


path = r'data/processed'
all_files = glob.glob(path + "/*.pt")
devices = [filename for filename in all_files]
index = 0
for filename in all_files:
    # if not filename.__contains__('dishwasher'):
    #     continue
    data = torch.load(f'{filename}')
    print('-------------------------------------------------------------------')
    print(filename.split('/')[-1].strip('.csv'))
    data.num_classes = len(data.y.unique())
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
    print(data.y)
    print(data)

    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)
    print(train_data, val_data, test_data)
    index += 1
    # pred = conventional_ml(train_data, test_data)
    #
    # plt.title("Predicted/ G-truth")
    # plt.plot(pred, label="pred")
    # plt.plot(test_data.y.view(-1, 1), label="g_truth", alpha=0.5)
    # plt.title(str(index))
    # plt.xlabel("timestep")
    # plt.ylabel("delta_p")
    # plt.legend()
    # plt.show()

    # print('End Pip
    # exit()
    # exit()
    #
    from utils import mse

    y_true = data.y.cpu().detach().numpy()
    y_hat = np.mean(y_true)
    print(mse(np.array([y_hat] * y_true.shape[0]), y_true))

    # exit('By Marinos')
    model = GCN(hidden_channels=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    epochs = 20
    train_losses = []
    val_losses = []
    print(np.unique(data.y.view(-1, 1)))

    for epoch in range(1, 50):
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
    # plt.title("Predicted/ G-truth")
    plt.plot(results, label="pred")
    plt.plot(data.y.view(-1, 1), label="g_truth", alpha=0.5)
    plt.title(filename.split('/')[-1].strip('.csv'))
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
    print('------------------------------End Pipeline-----------------------------')
