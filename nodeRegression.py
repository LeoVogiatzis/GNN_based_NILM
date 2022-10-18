import glob

import matplotlib.pyplot as plt
# from Gnn_Models.model import GCN
# from Gnn_Models import model
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

from Dataset import gsp_dataset
from Embeddings.Auto_Encoder import pairwise_auto_encoder
from Embeddings.Node2Vec import node_representations
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import sys
import seaborn as sns
import os
from torchmetrics import MeanAbsolutePercentageError


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
        x = F.dropout(x, p=0.8, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)

        return x


def weighted_mse_loss(pred, target):
    weight = np.ones_like(target.numpy())
    weight[target == 0] = 1
    # if weight is None else weight[target].to(pred.dtype)
    weight = torch.tensor(weight).to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def train(model):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    # out = model(dataset.x, dataset.edge_index)  # Perform a single forward pass.
    out = model(train_data.x, train_data.edge_index)
    print(out)
    print(train_data.y)
    # loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    # loss = criterion(out, train_data.y.view(-1, 1))
    # import pdb;pdb.set_trace()
    # loss = (out - train_data.y.view(-1, 1))**2
    # loss[train_data.y.view(-1, 1)!=0] *= 100
    # loss = torch.mean(loss)
    # loss = weighted_mse_loss(out.squeeze(), train_data.y.squeeze())
    loss = criterion(out.squeeze(), train_data.y.squeeze())  # view(-1, 1))
    # loss = loss * sample_weight
    loss.backward(retain_graph=True)  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    # scheduler.step()
    return loss


def test(model):
    model.eval()
    out = model(test_data.x, test_data.edge_index)
    test_loss = criterion(out.squeeze(), test_data.y.squeeze())  # .y.view(-1, 1)
    # Derive ratio of correct predictions.
    return test_loss, out


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


def plots(data):
    import seaborn as sns
    plt.style.use('seaborn-darkgrid')
    # path3 = "/home/leonidas/PycharmProjects/GNN_based_NILM/Centralities/"
    # os.makedirs(path3 + f"{filename.split('/')[-1].strip('.pt')}")
    Centralities = ['Betweeness', 'Closeness', 'Pagerank', 'Eigenvector']
    for i in range(4):
        from collections import Counter
        c = Counter(data.x[:, i].cpu().detach().numpy())
        print(c)
        sns.histplot(data=data.x[:, i].cpu().detach().numpy(), bins=150)
        # sns.histplot(data=c, bins=150)
        plt.yscale('log')
        plt.title(Centralities[i], fontsize=13)
        plt.xlabel('Ranking', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.savefig('microwave_' + Centralities[i] + '.png')
        # plt.savefig(f'/home/leonidas/PycharmProjects/GNN_based_NILM/Centralities/my_test{i}.eps')
        plt.show()

        # plt.hist(data.x[:, i].cpu().detach().numpy(), bins=100)
        # plt.yscale('log')
        # plt.show()
    # exit()


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
    base_model.fit(train_data.x.detach().numpy(), train_data.y.detach().numpy().ravel())
    base_accuracy = evaluate(base_model, train_data.x.detach().numpy(), train_data.y.detach().numpy().ravel())
    return base_model.predict(test_data.x.detach().numpy()).reshape(-1)
    # regr = RandomForestRegressor(random_state=0)
    #
    # CV_regr = GridSearchCV(estimator=regr, param_grid=param_grid,
    #                        cv=5, n_jobs=-1, verbose=2, return_train_score=True)
    #
    # with warnings.catch_warnings(record=True) as w:
    #     try:
    #         CV_regr.fit(train_data.x.detach().numpy(), train_data.y.detach().numpy().ravel())
    #     except ValueError:
    #         pass
    #     # print(repr(w[-1].message))
    # # train_data.x.detach().numpy(), train_data.y.detach().numpy().ravel()
    # # CV_regr.fit(np.array(train_data.x), np.array(train_data.y).ravel())
    # print(f'best parameters: {CV_regr.best_params_}')
    # best_grid = CV_regr.best_estimator_
    #
    # grid_accuracy = evaluate(best_grid, test_data.x.detach().numpy(), test_data.y.detach().numpy().ravel())
    # print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))
    #
    # mse = mean_squared_error(np.array(test_data.y.detach().numpy().ravel()), best_grid.predict(test_data.x.detach().numpy()))  # .reshape(-1, 1)
    # mae = mean_absolute_error(test_data.y.detach().numpy().ravel(), best_grid.predict(test_data.x.detach().numpy()))
    # print(f'best_estimator {best_grid}')
    # print(f'mse: {mse}')
    # print(f'mae: {mae}')
    # return best_grid.predict(test_data.x.detach().numpy()).reshape(-1)


#
# parser = argparse.ArgumentParser(description='Node Regression Pipeline')
# parser.add_argument('--Epochs', help='Insert  the epochs')
# parser.add_argument('--learning-rate', type=int,
#                     help='Insert (block height/Unix Epoch time)')
# parser.add_argument('--p', help='dropout')
#
# parser.add_argument('--time', type=int,
#                     help='Insert (block height/Unix Epoch time)')
# parser.add_argument('--public_key', help='Insert a public key')
# parser.add_argument('--time', type=int,
#                     help='Insert (block height/Unix Epoch time)')
# parser.add_argument('--public_key', help='Insert a public key')
# parser.add_argument('--time', type=int,
#                     help='Insert (block height/Unix Epoch time)')
#


# args = parser.parse_args()
# public_key = args.public_key
# absolute_param = args.time
#


path = r'data/median_filtering_Min_Max'
all_files = glob.glob(path + "/*.pt")
devices = [filename for filename in all_files]
index = 0
for filename in all_files:
    # if not filename.__contains__('microwave_6_house2.pt'):
    #     continue
    data = torch.load(f'{filename}')

    print('-------------------------------------------------------------------')
    print(filename.split('/')[-1].strip('.csv'))

    means, stds = data.x.mean(), data.x.std()
    data.x = (data.x - means) / stds

    # means, stds = data.y.mean(), data.y.std()
    # data.y = (data.y - means) / stds

    # plots(data)
    # continue

    data.num_classes = len(data.y.unique())
    # plots(data)
    # continue

    # v_min, v_max = data.y.min(), data.y.max()
    # new_min, new_max = -.1, .25
    # data.y = (data.y - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

    methods = ['features', 'Node2Vec', 'AE']
    # path2 = "/home/leonidas/PycharmProjects/GNN_based_NILM/Test_medd2/"
    # os.makedirs(path2 + f"{filename.split('/')[-1].strip('.pt')}")
    data.y = data.y.type(torch.FloatTensor)
    for embedding_method in methods:
        if embedding_method == 'Node2Vec':
            # continue
            embeddings = node_representations(data)
            data.x = embeddings.data

        elif embedding_method == 'AE':
            # continue
            data = pairwise_auto_encoder(data)
        else:
            # continue
            print(data.x)

        data.y = data.y.type(torch.FloatTensor)

        print(data.x)
        print(data.y)
        print(data)
        # plots(data)
        # continue
        # transform = RandomNodeSplit()
        # data = transform(data)

        transform = RandomLinkSplit(is_undirected=True)
        train_data, val_data, test_data = transform(data)
        print(train_data, val_data, test_data)
        index += 1

        pred = conventional_ml(train_data, test_data)
        # plt.rcParams["figure.figsize"] = (15, 6)
        sns.set_theme()
        plt.title("Predicted/ G-truth")
        plt.plot(pred, label="pred")
        plt.plot(test_data.y.view(-1, 1), label="g_truth", alpha=0.5)
        # plt.title(str(index))
        plt.xlabel("timestep")
        plt.ylabel("delta_p")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(path2 + f"{filename.split('/')[-1].strip('.pt')}" + "/" + embedding_method + '.png')
        plt.show()

        from utils import mse

        # exit()
        print(mse(np.array(test_data.y.view(-1, 1)), pred))

        y_true = data.y.cpu().detach().numpy()
        y_hat = np.mean(y_true)
        print(mse(np.array([y_hat] * y_true.shape[0]), y_true))
        from sklearn import metrics
        mean_abs_percentage_error = MeanAbsolutePercentageError()

        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(np.array(test_data.y.view(-1, 1)), pred))
        print('Mean Squared Error (MSE):', metrics.mean_squared_error(np.array(test_data.y.view(-1, 1)), pred))
        print('Root Mean Squared Error (RMSE):',
              np.sqrt(metrics.mean_squared_error(np.array(test_data.y.view(-1, 1)), pred)))
        mape = np.mean(np.abs((np.array(test_data.y.view(-1, 1)) - pred) / np.abs(np.array(test_data.y.view(-1, 1)))))
        print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
        print('Accuracy:', round(100 * (1 - mape), 2))
        print(mse(np.array(test_data.y.view(-1, 1)), pred))
        print('Random Forest')


        continue
        # model = GCN(hidden_channels=4)
        model = GCN(in_channels=4, hidden_channels=4, out_channels=1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0, last_epoch=-1, verbose=False)
        criterion = torch.nn.MSELoss()
        epochs = 20
        train_losses = []
        val_losses = []
        print(np.unique(data.y.view(-1, 1)))

        for epoch in range(1, 500):
            loss = train(model)
            test_loss, out = test(model)
            train_losses.append(loss.item())
            val_losses.append(test_loss.item())
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        print(model)



        results = model(data.x, data.edge_index)
        results = results.detach().numpy().reshape(-1)
        print(results)

        plt.plot(results, label="g_truth", alpha=0.5)
        plt.plot(data.y.view(-1, 1), label="g_truth", alpha=0.5)
        plt.title('Prediction/Ground Truth-Test')
        plt.xlabel("timestep")
        plt.ylabel("delta_p")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(path2 + f"{filename.split('/')[-1].strip('.pt')}" + "/" + embedding_method + 'Train' + '.png')
        plt.show()
        # # plt.savefig('foo.png')
        # continue
        # plt.show()

        plt.title("Predicted/ G-truth")
        plt.plot(out.detach().numpy().reshape(-1), label="pred")
        plt.plot(test_data.y.view(-1, 1), label="g_truth", alpha=0.5)
        # plt.title(filename.split('/')[-1].strip('.csv'))
        plt.xlabel("timestep")
        plt.ylabel("delta_p")
        plt.legend()
        # plt.savefig(path2 + f"{filename.split('/')[-1].strip('.pt')}" + "/" + embedding_method + 'Pre_G_truth' + '.png')
        # plt.savefig('foo.png')

        plt.show()

        # plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(val_losses, label="test")
        plt.plot(train_losses, label="train")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(path2 + f"{filename.split('/')[-1].strip('.pt')}" + "/" + embedding_method + 'loss' + '.png')
        plt.show()
        print('------------------------------End Pipeline-----------------------------')
