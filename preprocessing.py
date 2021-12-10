import warnings

import pandas as pd
from IPython.display import display

pd.options.display.max_columns = 20
warnings.filterwarnings("ignore")
import glob
from sklearn import preprocessing
import networkx as nx


def read_label():
    label = {}
    for i in range(1, 7):
        hi = 'low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


labels = read_label()
for i in range(1, 3):
    print('House {}: '.format(i), labels[i], '\n')


def read_merge_data(house):
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
    df.drop(['uhonix_time', 'timestamp'], axis=1, inplace=True)
    return df


def standardization(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns.to_list())
    return df


df = {}
for i in range(1, 7):
    df[i] = read_merge_data(i)

for i in range(1, 3):
    print('House {} data has shape: '.format(i), df[i].shape)
    display(df[i].tail(3))

house1 = df[1].diff().dropna()
house2 = df[2].diff().dropna()
house3 = df[3].diff().dropna()
house4 = df[4].diff().dropna()
house5 = df[5].diff().dropna()
house6 = df[6].diff().dropna()

house1 = standardization(house1)
print(house1.head)
print('-----------------------------')
print(house1.tail)
house1.to_csv('data/house1')
exit()
edgelist = pd.DataFrame(columns=['source', 'destination', 'weight'])
edgelist['source'] = house1.index
edgelist['destination'] = house1.index + 1
edgelist['weight'] = house1['mains_1']

G = nx.from_pandas_edgelist(df=edgelist, source='source', target='destination', edge_attr=True,
                            create_using=nx.Graph(name='Travian_Graph'))
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
