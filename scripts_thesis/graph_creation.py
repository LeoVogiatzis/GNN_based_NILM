import glob

import matplotlib.pyplot as plt
# from Gnn_Models.model import GCN
# from Gnn_Models import model
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import medfilt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
import sys
import pandas as pd
import torch_geometric.transforms as T

sys.path.append('../')

from Dataset import gsp_dataset
from Embeddings.Auto_Encoder import pairwise_auto_encoder
from Embeddings.Node2Vec import node_representations
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
        'max_depth': [3, 6, 9, 12, 16, 30, 50, 80, 90, 100, 110],
        'max_samples': [0.5, 0.75, 0.95],
        'max_features': [2, 3, 4],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    }

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
    print(CV_regr.best_params_)
    best_grid = CV_regr.best_estimator_
    grid_accuracy = evaluate(best_grid, np.array(train_data.x), np.array(train_data.y).ravel())
    print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

    mse = mean_squared_error(np.array(test_data.y), best_grid.predict(np.array(test_data.x)))  # .reshape(-1, 1)
    mae = mean_absolute_error(np.array(test_data.y), best_grid.predict(np.array(test_data.x)))
    print(mse)
    print(mae)
    print(best_grid)
    return best_grid.predict(test_data.x.detach().numpy()).reshape(-1)


def median_filter(x):
    print(x)
    x = x.to_numpy()
    y = medfilt(x)
    return y


def standardization(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns.to_list())
    return df


def standar_scaler(df):
    from sklearn.preprocessing import StandardScaler
    trans = StandardScaler(df)
    x_scaled = trans.fit_transform(df)
    df = pd.DataFrame(x_scaled, columns=df.columns.to_list())
    return df


df = pd.read_csv('../data/House.csv',
                 usecols=['Unnamed: 0', 'kitchen_outlets_house_2_x', 'lighting_house_2', 'refrigerator_house_2',
                          'microwave_house_2'])
df = df.set_index(pd.DatetimeIndex(df['Unnamed: 0']))
print(df.head(1))
print(df.shape[0])
df = df.resample('1T').sum()
print(df.shape[0])

df.transform(lambda x: median_filter(x))
df = standardization(df)


def write_appliance_files(df):
    for i in df.columns:
        df[i].to_frame().to_csv('../data/raw/' + str(i) + '.csv', index=False)


write_appliance_files(df)
path = r'./data/raw'
all_files = glob.glob(path + "/*.csv")
houses = [filename for filename in all_files]
index = 0
for filename in all_files:
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomNodeSplit(),
    ])
    file = filename.split('/')[3]
    dataset = gsp_dataset.NilmDataset(root='data', filename=f'{file}', window=20, sigma=20, transform=transform)

    # if index > 0:
    #     index += 1
    data = dataset[0]
    print(data)

    data.y = data.y.type(torch.FloatTensor)
    print(data.x)

    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)
    print(train_data, val_data, test_data)
    index += 1
    appliance = filename.split('/')[-1].strip('.csv')
    torch.save(data, f'../data/processed/{appliance}.pt')
