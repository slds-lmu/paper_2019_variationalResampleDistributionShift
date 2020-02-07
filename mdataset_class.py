"""Subset a dataset with different methods"""
import os
import torchvision
import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from data_manipulator import concatenate_data_from_dir

class SubDatasetByIndices(Dataset):
    def __init__(self, dataset_name, indices, transform=None):
        tv_method = getattr(torchvision.datasets, dataset_name)
        trainset_temp = tv_method(root='./data', train=True, download=True, transform=transform)
        testset_temp = tv_method(root='./data', train=False, download=False, transform=transform)
        trte_ds = ConcatDataset((trainset_temp, testset_temp))
        self.subset = torch.utils.data.Subset(trte_ds, indices)

    def __len__(self):
        return self.subset.__len__()

    def __getitem__(self, idx):
        return self.subset.__getitem__(idx)

class TrTeData():
    def __init__(self, dataset_name, transform=None):
        tv_method = getattr(torchvision.datasets, dataset_name)
        trainset_temp = tv_method(root='./data', train=True, download=True, transform=transform)
        testset_temp = tv_method(root='./data', train=False, download=False, transform=transform)
        trte = ConcatDataset((trainset_temp, testset_temp))
        self.data = trte

class SubdomainDataset(Dataset):
    """Torch Dataset gathering data from input subdomain indice"""
    def __init__(self, config_volatile, transform=None, list_idx=[0]):
        """
        Args:
            config_volatile (module): python module with configurations written down
            transform (callable, optional): Optional transform to be applied on a sample. See Torch documentation
            list_idx (list): the list of indexes of the cluster to choose as trainset or testset, for example
            trainset = SubdomainDataset(list_idx = [0,1, 2, 3])
            testset = SubdomainDataset(list_idx = [4])
        """
        self.root_dir = os.path.join(config_volatile.rst_dir, config_volatile.data_path)
        self.pattern = config_volatile.global_index_name
        self.transform = transform
        if not tf.gfile.Exists(os.path.join(self.root_dir, self.pattern)):
            _, self.global_index = concatenate_data_from_dir(self.root_dir, num_labels=config_volatile.num_labels, num_clusters=config_volatile.num_clusters)
        else:
            self.global_index = np.load(os.path.join(self.root_dir, self.pattern), allow_pickle=True)
            # global_index example:{'0': [15352, 21, ..], '1':[1121, 3195,...]}
        self.list_idx = list_idx
        all_inds = []
        print('cluster index list:' + str(list_idx))
        for index in self.list_idx:  # iterate all **chosen** clusters/subdomains
            to_append = self.global_index.get(str(index))
            to_append = self.global_index.get(str(index))
            print('\n size of cluster:' + str(np.shape(to_append)) + '\n')
            all_inds = np.append(all_inds, to_append)
            print(all_inds.shape)
        self.all_inds = all_inds.tolist()
        self.all_inds = [round(x) for x in self.all_inds] # make to be integer
        trte = TrTeData(config_volatile.dataset_name)
        self.subset = torch.utils.data.Subset(trte.data, self.all_inds)

    def __len__(self):
        return self.subset.__len__()

    def __getitem__(self, idx):
        return self.subset.__getitem__(idx)
