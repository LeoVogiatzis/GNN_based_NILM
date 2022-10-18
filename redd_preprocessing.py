import warnings
import pandas as pd
from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# pd.options.display.max_columns = 20
warnings.filterwarnings("ignore")
import glob
from sklearn import preprocessing
import networkx as nx
import numpy as np

global label
import argparse
import os.path
from os import path
from itertools import combinations
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import distance
import scipy
import itertools


def read_label():
    label = {}
    for i in range(1, 7):
        hi = './low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0] + '_house' + str(i)
    return label


labe[data.test_mask]
ls = read_label()
for i in range(1, 7):
    print('House {}: '.format(i), labels[i], '\n')


def read_merge_data(house):
    path = './low_freq/house_{}/'.format(house)
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


def write_appliance_files(df):
    for i in df.columns:
        df[i].to_csv('./data/3T/' + str(i) + '.csv', index=False)


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


def cyclical_encoding():
    import math

    df[i]["x_norm"] = 2 * math.pi * df[i]["kitchen_outlets_house_2_x"] / df[2]["kitchen_outlets_house_2_x"].max()

    df["cos_x"] = np.cos(df[2]["x_norm"])


df = {}
for i in range(1, 7):
    df[i] = read_merge_data(i)

for i in range(1, 3):
    print('House {} data has shape: '.format(i), df[i].shape)
    display(df[i].tail(3))

# # df = {}
# for i in range(1, 7):
house2 = df[2].transform(lambda x: median_filter(x))
house2 = standardization(house2)
house2.to_csv('/home/leonidas/PycharmProjects/GNN_based_NILM/data/House.csv')
exit()
# # write_appliance_files(df[i])
# x = 1
# exit()
for i in df.keys():
    print(df[i].shape[0])
    df[i] = df[i].resample('1T').sum()
    print(df[i].shape[0])
    df[i].transform(lambda x: median_filter(x))
    df[i] = standardization(df[i])
    write_appliance_files(df[i])
