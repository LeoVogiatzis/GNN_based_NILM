#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# %%


def graph_creation(event, delta_p, sigma):
    winL = 1000  # this defines  number of observations in a window, the algorthm works in a sliding window manner
    Smstar = np.zeros((len(event), 1))
    for k in range(0, int(np.floor(len(event) / winL))):
        r = []
        event_1 = event[k * winL:((k + 1) * winL)]
        # followed as such from the MATLAB code
        r.append(delta_p[event[0]])
        [r.append(delta_p[event_1[i]]) for i in range(0, len(event_1))]
        templen = winL + 1
        Sm = np.zeros((templen, 1))
        Sm[0] = 1;

        Am = np.zeros((templen, templen))
        for i in range(0, templen):
            for j in range(0, templen):
                Am[i, j] = math.exp(-((r[i] - r[j]) / sigma) ** 2);
                # Gaussian kernel weighting function
        Dm = np.zeros((templen, templen));
        # create diagonal matrix
        for i in range(templen):
            Dm[i, i] = np.sum(Am[:, i]);
        Lm = Dm - Am;
        Smstar[k * winL:(k + 1) * winL] = np.matmul(np.linalg.pinv(Lm[1:templen, 1:templen]),
                                                    ((-Sm[0].T) * Lm[0, 1:templen]).reshape(-1, 1));
        x = 1
        return Smstar, Lm, Dm, Am


def dataset_preprocessing():
    df.index = pd.to_datetime(df.index)
    dfd = pd.read_csv(csvfiledisaggr, index_col="Time")  # read file with ground truth disaggregated appliances
    dfd.index = pd.to_datetime(dfd.index)

    # select date range
    start_date = '2011-04-23'  # from 2011-04-23
    end_date = '2011-05-02'  # to 2011-05-01
    mask = (df.index > start_date) & (df.index < end_date)
    df = df.loc[mask]
    mask = (dfd.index > start_date) & (dfd.index < end_date)
    dfd = dfd.loc[mask]


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
    main_val = df[['dishwaser_20']].values  # get only readings
    # main_ind = df.index  # get only timestamp
    data_vec = main_val
    # signature_database = "signature_database_labelled.csv"  # the signatures were extracted of power analysis from April 28th to 30th
    threshold = 2000  # threshold of DTW algorithm used for appliance power signature matching

    delta_p = [np.round(data_vec[i + 1] - data_vec[i], 2) for i in range(0, len(data_vec) - 1)]
    event = [i for i in range(0, len(delta_p))]

    # event = [i for i in range(0, len(delta_p)) if (delta_p[i] > T_Positive or delta_p[i] < T_Negative)]
    Smstar, Lm, Dm, Am = graph_creation(event, delta_p, sigma)


if __name__ == '__main__':
    main()
