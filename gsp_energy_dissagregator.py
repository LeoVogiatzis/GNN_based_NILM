#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def graph_creation(data_vec, sigma,
                   window=1000):  # this defines  number of observations in a window, the algorthm works in a sliding window manner
    data_aggr = []

    for k in range(0, int(np.floor(len(data_vec) / window))):
        data_aggr.append(np.mean(data_vec[k * window:((k + 1) * window)]))
    
    if (len(data_vec) % window > 0):
        data_aggr.append(np.mean(data_vec[int(np.floor(len(data_vec) / window)) * window:]))
    delta_p = [np.round(data_aggr[i + 1] - data_aggr[i], 2) for i in range(0, len(data_aggr) - 1)]
    Am = np.zeros((len(delta_p), len(delta_p)))
    for i in range(0, Am.shape[0]):
        for j in range(0, Am.shape[1]):
            Am[i, j] = math.exp(-((delta_p[i] - delta_p[j]) / sigma) ** 2)
    return Am


def main():
    # G = nx.read_gpickle('graphs/microwave_3.gpickle')
    house5 = pd.read_csv('data/house5.csv')

    df = pd.read_csv('data/house5.csv')  # read demo file with aggregated active power

    # Please read the paper to undertand following parameters. Note initial values of these parameters depends on the appliances used and the frequency of usage.
    sigma = 20;
    ri = 0.15
    T_Positive = 20;
    T_Negative = -20;
    # Following parameters alpha and beta are used in Equation 15 of the paper
    # alpha define weight given to magnitude and beta define weight given to time
    alpha = 0.5
    beta = 0.5
    # this defines the  minimum number of times an appliance is set ON in considered time duration
    instancelimit = 3

    # %%
    main_val = df['dishwaser_20'].values  # get only readings
    # main_ind = df.index  # get only timestamp
    data_vec = main_val
    # signature_database = "signature_database_labelled.csv"  # the signatures were extracted of power analysis from April 28th to 30th
    threshold = 2000  # threshold of DTW algorithm used for appliance power signature matching

    Am = graph_creation(data_vec, sigma, window=77)
    x = 1


if __name__ == '__main__':
    main()
