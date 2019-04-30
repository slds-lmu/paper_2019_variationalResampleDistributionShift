from PIL import Image
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import tensorflow as tf
import numpy as np
import os, sys
os.getcwd()
import os.path
print(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

#print('The value of dataset_name could only be: {}'.format("mnist or fashion-mnist"))

import utils_parent

# FIXME: the problem is the folder above has the same module named utils
from data_generator import concatenate_data_from_dir
#sys.path.pop(0)  # remove parent folder from search path
#os.path.realpath(__file__)
#sys.path.append(os.path.dirname(os.path.dirname()))

#data_path = '../results/VAE_fashion-mnist_64_62'
#result_path = '../results/VAE_fashion-mnist_64_62'
#if not tf.gfile.Exists(data_path+"/global_index_cluster_data.npy"):
#    _,global_index = concatenate_data_from_dir(data_path,num_labels=num_labels,num_clusters=num_cluster)
#else:global_index = np.load(data_path+"/global_index_cluster_data.npy",allow_pickle=True)
## type(global_index)  numpy.ndarray
#len(global_index.item().get(str(1)))
#
#a = global_index.item().get(str(0))
#b = global_index.item().get(str(1))
#np.append(a, b)
#np.concatenate((a, b))
#gen = (a for a in [[2,4], [1,5,7]])
#print(list(gen))  # can only be executed once
#import sys
#map(sys.stdout.write, gen)
#from itertools import chain
#gen2 = chain(gen)
#list(gen2).ravel()
#import config
### smaller data, for debug, but this smaller data should find its intersection with the z data
#X, y = utils_parent.load_mnist("fashion-mnist")
#from sklearn.model_selection import train_test_split
#X_1, X_2, Y_1, Y_2 = train_test_split(X, y, stratify=y, test_size=0.01)
#y.take([1], axis = 0).shape
#y[[1,2,3], ]
#X[1,]
#X[1].shape
#
class VGMMDataset(Dataset):
    """Dataset after VGMM clustering"""
    def __init__(self, pattern = "/global_index_cluster_data.npy", root_dir = '../results/VAE_fashion-mnist_64_62', transform=None, list_idx = [0], dsname = "fashion-mnist", num_labels = 10, num_cluster = 5,cluster = True,index=[]):
        """
        Args:
            pattern (string): Path to the npy file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            list_idx (list): the list of indexes of the cluster to choose as trainset or testset
        """
        X, y = utils_parent.load_mnist(dsname)
        y = y.argmax(axis=1)
        self.root_dir = root_dir
        self.pattern = pattern
        self.transform = transform
        if cluster ==True:

            if not tf.gfile.Exists(self.root_dir + self.pattern):
                _, self.global_index = concatenate_data_from_dir(self.root_dir, num_labels=num_labels,
                                                                 num_clusters=num_cluster)
            else:
                self.global_index = np.load(self.root_dir + pattern, allow_pickle=True)
            self.list_idx = list_idx
            all_inds = []
            for index in self.list_idx:
                to_append = self.global_index.item().get(str(index))
                all_inds = np.append(all_inds, to_append)
            self.all_inds = all_inds.tolist()
            # self.all_inds = map(round, self.all_inds)
            self.all_inds = [round(a) for a in self.all_inds]
            self.samples = {"x": X.take(self.all_inds, axis=0), "y": y.take(self.all_inds, axis=0)}
        else:
            self.all_inds = index
            self.samples = {"x": X[index], "y": y[index]}



    def __len__(self):
        return len(self.all_inds)

    def __getitem__(self, idx):
        x, y = self.samples["x"][idx, ], self.samples["y"][idx, ]
        x = Image.fromarray(x.squeeze(), mode = 'L')
        if self.transform:
            #x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x,y


class WVGMMDataset(VGMMDataset):
    def __init__(self, conf_manager, list_idx):
        super(WVGMMDataset, self).__init__(pattern = conf_manager.pattern, root_dir = conf_manager.root_dir, list_idx = list_idx)

if __name__ == '__main__':

    # vgmmds = VGMMDataset()
    trainset = VGMMDataset(list_idx = [0,1, 2, 3])
    testset = VGMMDataset(list_idx = [4])
