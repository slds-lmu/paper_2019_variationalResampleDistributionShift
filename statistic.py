import seaborn as sns
import numpy as np
import json

#load data
y = np.load('/Users/wangyu/Documents/LMU/Fashion_mnist/mycode/results/VAE_fashion-mnist_64_62/y.npy')
def load_dict(path):
    with open(path) as f:
        my_dict = json.load(f)
    return my_dict
# cluster_dict = load_dict('/Users/wangyu/Documents/LMU/Fashion_mnist/mycode/results/VAE_fashion-mnist_64_62/cluster_dict.json')
cluster_dict = load_dict('/Users/wangyu/Documents/LMU/Fashion_mnist/mycode/results/VAE_fashion-mnist_64_62/pos_index_cluster.json')

y_0 = y[cluster_dict['0']]
y_0 = np.sum(y_0,axis = 0)

y_1 = y[cluster_dict['1']]
y_1 = np.sum(y_1,axis = 0)


y_2 = y[cluster_dict['2']]
y_2 = np.sum(y_2,axis = 0)

y_3 = y[cluster_dict['3']]
y_3 = np.sum(y_3,axis = 0)

y_4 = y[cluster_dict['4']]
y_4 = np.sum(y_4,axis = 0)
print(y_0)
print(y_1)
print(y_2)
print(y_3)
print(y_4)