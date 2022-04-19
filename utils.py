import numpy as np


def mae(prediction, true):
    MAE = abs(true - prediction)
    MAE = np.sum(MAE)
    MAE = MAE / len(prediction)
    return MAE


def mse(prediction, true):
    MSE = (true - prediction) ** 2
    MSE = np.sum(MSE)
    MSE = MSE / len(prediction)
    return MSE


def sae(prediction, true, N):
    T = len(prediction)
    K = int(T / N)
    SAE = 0
    for k in range(1, N):
        pred_r = np.sum(prediction[k * N: (k + 1) * N])
        true_r = np.sum(true[k * N: (k + 1) * N])
        SAE += abs(true_r - pred_r)
    SAE = SAE / (K * N)
    return SAE


def standardize_data(data, mu=0.0, sigma=1.0):
    data -= mu
    data /= sigma
    return data


def normalize_data(data, min_value=0.0, max_value=1.0):
    data -= min_value
    data /= max_value - min_value
    return data
