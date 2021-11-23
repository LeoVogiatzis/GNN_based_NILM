#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import pandas as pd
from IPython.display import display
import networkx as nx
import time

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
from itertools import combinations
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import distance
import scipy
import itertools


def graph_creation(device):
    nodes = pd.DataFrame(columns=['drift', 'Timestamp'])
    nodes['drift'] = device.diff().shift(-1).iloc[:-1][:10000]
    threshold = nodes['drift'].abs().mean()
    nodes['Timestamp'] = nodes.index
    nodes.reset_index(inplace=True)
    nodes = nodes[abs(nodes['drift']) > threshold]
    # edgelist.reset_index(inplace=True)
    edgelist = pd.DataFrame(
        [(x[0], x[1],
          np.exp(-(4 * np.log(2) * (np.linalg.norm(nodes['drift'][x[0]] - nodes['drift'][x[1]]))) ** 10) / 2 ** 2) for x
         in
         itertools.combinations(nodes['drift'].index, 2)],
        columns=['source', 'destination', 'gaussian_kernel'])
    edgelist = edgelist.loc[(edgelist[['gaussian_kernel']] != 0).all(axis=1)]
    # edgelist.drop_duplicates(inplace=True)
    nodes["state"] = np.where(nodes["drift"] > 0, 1, 0)
    G = nx.from_pandas_edgelist(df=edgelist, source='source', target='destination', edge_attr=True,
                                create_using=nx.Graph(name='Nilm_Graph'))

    for node in G.nodes():
        G.nodes[node]['Timestamp'] = nodes['Timestamp'][node]
        G.nodes[node]['state'] = nodes['state'][node]
        G.nodes[node]['drift'] = nodes['drift'][node]

    # for index, row in nodes.iterrows():
    #     print(device)
    #     # print(row['Timestamp'])
    #     # print(row['state'])
    #     print(f'Add node features {device.name}')
    #     G.nodes[row['drift']]['Timestamp'] = row['Timestamp']
    #     G.nodes[row['drift']]['state'] = row['state']

    nx.write_graphml(G, 'graphs/' + str(device.name) + '.graphml')


def main():
    # G = nx.read_gpickle('graphs/microwave_3.gpickle')
    house5 = pd.read_csv('data/house5.csv')
    # house5.drop(columns=['Unnamed: 0'], inplace=True)
    print(house5.columns)
    # graph_creation(house5['lighting_4'])
    house5[house5.columns].apply(lambda x: graph_creation(x))


if __name__ == '__main__':
    main()
