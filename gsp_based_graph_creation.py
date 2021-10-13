import warnings

import pandas as pd
from IPython.display import display
import networkx as nx

pd.options.display.max_columns = 20
warnings.filterwarnings("ignore")
import glob
from sklearn import preprocessing
import networkx as nx
import numpy as np

global label
import argparse
import os.path
from os import path


def read_label(dataset):
    if dataset == 'low_freq':
        houses = 7
    elif dataset == 'UK-DALE':
        houses = 6
    else:
        houses = 21

    label = {}
    for i in range(1, houses):
        hi = dataset + '/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


def read_merge_data(house, labels, dataset):
    num_apps = 0
    if dataset == 'low_freq':
        houses = 7
    elif dataset == 'UK-DALE':
        houses = 6
        num_apps = -1
    else:
        houses = 21
    path = dataset + '/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'channel*')) + num_apps
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df


def gaussian(alt_df):
    return np.exp(-(4 * np.log(2) * alt_df ** 2) / 5 ** 2)


def main():
    # parsing arguments
    parser = argparse.ArgumentParser(description='Load Benchmark NIlM datasets and create a graph')
    parser.add_argument('--dataset', help='Insert the dataset')
    parser.add_argument('--house', type=int,
                        help='Insert the house of the dataset')
    parser.add_argument('--appliance', help='Insert the appliance that you want to create a graph')
    # parser.add_argument('--p2pkh', help='Insert a p2pkh')
    args = parser.parse_args()
    dataset = args.dataset
    house = args.house
    appliance = args.appliance

    # if path.exists('/graphs/'+ dataset + house + str(appliance)):
    #     exit('Try another Graph')


    dataset = 'low_freq'  # args.dataset

    if dataset != ' UK-REFIT':
        labels = read_label(dataset)
        for i in range(1, 3):
            print('House {}: '.format(i), labels[i], '\n')

    # data = pd.read_csv('UK-REFIT/House_1.csv')
    df = {}
    for i in range(1, 7):
        df[i] = read_merge_data(i, labels, dataset)
    # alt_df = df[1]['mains_1'].diff()

    edgelist = pd.DataFrame(columns=['source', 'destination', 'weight'])
    edgelist['source'] = df[1]['mains_1'].diff().shift(-1).iloc[:-1]
    # edgelist = edgelist[:-1]
    # edgelist['source'] = edgelist['source'].shift(-1)# .dropna()
    edgelist = edgelist.iloc[1:, :]
    edgelist.reset_index(inplace=True)

    timestamps = edgelist['source'].iloc[0::2].index
    temp = edgelist['source'].iloc[1::]
    # edgelist = edgelist[:-1]
    edgelist['destination'] = temp
    edgelist['destination'] = edgelist['destination'].shift(-1)
    # edgelist = pd.DataFrame(
    #     {'source': edgelist['source'].iloc[::2].values, 'destination': edgelist['source'].iloc[1::2].values})
    edgelist.set_index(timestamps, inplace=True)
    edgelist['gaussian_kernel'] = np.exp(
        -(4 * np.log(2) * (edgelist['source'] - edgelist['destination']) ** 2) / 5 ** 2)

    G = nx.from_pandas_edgelist(df=edgelist, source='source', target='destination', edge_attr=True,
                                create_using=nx.MultiDiGraph(name='Nilm_Graph'))
    nx.write_gpickle(G, '/graphs/'+ dataset + house + str(appliance))


if __name__ == '__main__':
    main()
