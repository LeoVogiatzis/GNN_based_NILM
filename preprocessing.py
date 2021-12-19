import networkx as nx
from sklearn import preprocessing
import glob
import warnings

import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

pd.options.display.max_columns = 20
warnings.filterwarnings("ignore")
import numpy as np


def read_label():
    label = {}
    for i in range(3, 5, 2):
        hi = 'low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip(
                ) + '_' + splitted_line[0]
    return label


def read_merge_data(house, labels):
    path = 'low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df


def plot_df(df, title):
    apps = df.columns.values
    num_apps = len(apps)
    fig, axes = plt.subplots((num_apps + 1) // 2, 2, figsize=(24, num_apps * 2))
    for i, key in enumerate(apps):
        axes.flat[i].plot(df[key], alpha=0.6)
        axes.flat[i].set_title(key, fontsize='15')
    plt.suptitle(title, fontsize='30')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.show()


def read_merge_features(house, labels):
    path = 'high_freq/house_{}/'.format(house)
    file = path + 'current_1.dat'
    df = pd.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'current*'))
    for i in range(2, num_apps + 1):
        file = path + 'current{}.dat'.format(i)
        data = pd.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df


def standardization(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns.to_list())
    return df


def main():
    labels = read_label()
    for i in range(3, 5, 2):
        print('house {}: '.format(i), labels[i], '\n')

    df = {}
    for i in range(3, 5, 2):
        df[i] = read_merge_data(i, labels)

    for i in range(3, 5, 2):
        print('house {} data has shape: '.format(i), df[i].shape)
        display(df[i].tail(3))

    dates = {}
    for i in range(3, 5, 2):
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(
            i, len(dates[i]), dates[i][0], dates[i][-1]))
        print(dates[i], '\n')

    df_currents = pd.read_csv('high_freq/house_3/current_1.dat', sep=' ', header=None)

    for i in range(3, 5, 2):
        plot_df(df[i].loc[:dates[i][1]], 'First 2 day data of house {}'.format(i))

    # Plot total energy sonsumption of each appliance from two houses
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    plt.suptitle('Total enery consumption of each appliance', fontsize=30)
    cons1 = df[3][df[3].columns.values[2:]].sum().sort_values(ascending=False)
    app1 = cons1.index
    y_pos1 = np.arange(len(app1))
    axes[0].bar(y_pos1, cons1.values, alpha=0.6)
    plt.sca(axes[0])
    plt.xticks(y_pos1, app1, rotation=45)
    plt.title('House 3')
    plt.show()

    # exit('test')


if __name__ == '__main__':
    main()

# house1 = df[1].diff().dropna()
# house2 = df[2].diff().dropna()
# house3 = df[3].diff().dropna()
# house4 = df[4].diff().dropna()
# house5 = df[5].diff().dropna()
# house6 = df[6].diff().dropna()

# house1 = standardization(house1)
# print(house1.columns)
# print(house1.head)
# print('-----------------------------')
# print(house1.tail)
# print(house1.describe)
# exit()
# edgelist = pd.DataFrame(columns=['source', 'destination', 'weight'])
# edgelist['source'] = house1.index
# edgelist['destination'] = house1.index + 1
# edgelist['weight'] = house1['mains_1']

# G = nx.from_pandas_edgelist(df=edgelist, source='source', target='destination', edge_attr=True,
#                             create_using=nx.Graph(name='Travian_Graph'))
# houses = {}
# h_key = 0
# for k, v in df.items():
#     h_key =+1
#     houses['house' + str(h_key)].update(df[k].diff().dropna())

# df[1].set_index('timestamp').diff()

# data = pd.concat(df, axis=1).sum(axis=1, level=0)
# print(data.head())
# x=1
# def main():
#     pass
