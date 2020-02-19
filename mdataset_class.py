"""Subset a dataset with different methods"""
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import tensorflow as tf

import utils_parent
from config_manager import ConfigManager

def dataset_name_tr(dataset_name):
    if dataset_name == 'fashion-mnist': dataset_name = "FashionMNIST"
    elif dataset_name =='cifar10': dataset_name = "CIFAR10"
    return dataset_name

class Data2ResampleBase(object):
    def init_load(self, *args, **kwargs):  # *args, **kwargs for subclass to override with custom parameters
        raise NotImplementedError
    def get_subdomain_data(self):
        pass

class InputDataset(Data2ResampleBase):
    def init_load(self, *args, **kwargs):
        pass

    def prepare_data(self, config_parent, args,train_eval_list,test_list,resize, method, transform_train, transform_test):
        # Data Uplaod
        debug = args.debug
        return self.split_te_tr_val(config_parent, method, train_eval_list, test_list, transform_train, transform_test, debug)

    def split_te_tr_val(self, config_volatile, method, train_eval_list, test_list, transform_train, transform_test, debug = False):
        debug_frac = ConfigManager.debug_subset_frac
        train_val_frac = ConfigManager.train_val_frac
        dataset_name = config_volatile.dataset_name
        if method == "vgmm":
            train_eval_set = SubdomainDataset(config_volatile = config_volatile,  transform=transform_train, list_idx=train_eval_list)
            testset = SubdomainDataset(config_volatile = config_volatile, transform=transform_test, list_idx=test_list, )
        elif method == "rand":
            train_eval_set = SubDatasetByIndices(dataset_name= dataset_name, indices=train_eval_list, transform=transform_train)
            testset = SubDatasetByIndices(dataset_name = dataset_name, indices=test_list, transform=transform_test)
        else:
            raise NotImplementedError
        outputs = train_eval_set.num_classes
        inputs = train_eval_set.num_channels
        if debug:
            small_size = int(debug_frac*len(train_eval_set))
            drop_size = len(train_eval_set)-small_size
            train_eval_set,_ = torch.utils.data.random_split(train_eval_set, [small_size, drop_size])

            small_size = int(debug_frac * len(testset))
            drop_size = len(testset) - small_size
            testset, _ = torch.utils.data.random_split(testset, [small_size, drop_size])
        train_size = int(train_val_frac * len(train_eval_set))
        eval_size = len(train_eval_set) - train_size
        trainset, evalset = torch.utils.data.random_split(train_eval_set, [train_size, eval_size])   # split train_eval_set into trainset and evalset
        return trainset, evalset, testset, inputs, outputs

    @staticmethod
    def split_data_according_to_label(z, y, num_labels):
        d = {}
        for i in range(num_labels):
            # d[i]: represent the index of data with label i
            d[str(i)] = np.where(y[:, i] == 1)[0]
        return d

    @staticmethod
    def concatenate_data_from_dir(config_volatile):
        pos = {}  # pos[i_cluster] correspond to the z value (concatenated) of cluster i_cluster
        global_index = {}  # global_index['cluster_1'] correspond to the global index with respect to the original data of cluster 1

        if isinstance(config_volatile, ConfigManager):
            data_path = config_volatile.get_data_path()
        else:
            data_path = config_volatile.data_path
        data_path, num_labels, num_clusters = data_path, config_volatile.num_labels, config_volatile.num_clusters
        for i_label in range(num_labels):
            path = data_path + ConfigManager.label_name + str(i_label)   #FIXME! $"/L"
            path = os.path.join(config_volatile.rst_dir, path)
            z = np.load(path + config_volatile.z_name)  # z = np.load(path + "/z.npy")
            y = np.load(path + config_volatile.y_name)  # y is the index dictionary with respect to global data
            cluster_predict = np.load(path + config_volatile.cluster_predict_npy_name)
            if i_label == 0:  # initialize the dictionary, using the first class label for each key of the dictionary, where key is the cluster index
                for i_cluster in range(num_clusters):
                    pos[str(i_cluster)] = z[np.where(cluster_predict == i_cluster)]
                    global_index[str(i_cluster)] = y[np.where(cluster_predict == i_cluster)]
            else:
                for i_cluster in range(num_clusters):
                    pos[str(i_cluster)] = np.concatenate((pos[str(i_cluster)], z[np.where(cluster_predict == i_cluster)]))
                    global_index[str(i_cluster)] = np.concatenate((global_index[str(i_cluster)], y[np.where(cluster_predict == i_cluster)]))
        return pos, global_index


    def __init__(self, dataset_name, label, num_labels):
        self.dataset_name = dataset_name
        # if flag labeled is true, train data is the subset of data(Mnist) which has same label
        X, y = self.load_torchvision_data2np(self.dataset_name,)
        if label != -1:
            # dict[i] represent data index with label i
            mdict4gind = InputDataset.split_data_according_to_label(X, y, num_labels)
            # extract data with label i from global training data
            self.data_X = X[mdict4gind[str(label)]]
            # y represent the index with label i
            self.data_y = y[mdict4gind[str(label)]]
            self.data_gind = mdict4gind[str(label)]
        else:
            self.data_X, self.data_y = X, y

    def load_torchvision_data2np(self, dataset_name = "CIFAR10", num_classes = 10, shuffle=False, seed=547, allowed_input_channels = [1, 3]):
        """This looks like bad code since we are not using Dataloader here, but access the data directly Dataset.data, however, since our data need to be feed to tensorflow, we have to make them  numpy array"""
        dataset_name = dataset_name_tr(dataset_name)
        tv_method = getattr(torchvision.datasets, dataset_name)
        # train = True
        # function transform is defined in this module as a hook to torchvision
        trainset_temp = tv_method(root='./data', train=True, download=True, transform=utils_parent.transform)  # FIXME: useless transform here
        trX = trainset_temp.data
        if(trX.shape[-1] != allowed_input_channels[0] and trX.shape[-1] != allowed_input_channels[1]): trX = trX.unsqueeze(-1)
        trY = trainset_temp.targets
        # train = False
        testset_temp = tv_method(root='./data', train=False, download=False, transform=utils_parent.transform)
        teX = testset_temp.data
        if(teX.shape[-1] != allowed_input_channels[0] and trX.shape[-1] != allowed_input_channels[1]): teX = teX.unsqueeze(-1)
        teY = testset_temp.targets
        # torch
        cd = ConcatDataset((trainset_temp, testset_temp))
        #return cd.data, cd.targets
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)
        yy = np.zeros((len(y), num_classes))
        yy[np.arange(len(y)), y] = 1
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(X)
            np.random.seed(seed)
            np.random.shuffle(yy)
        return X/255., yy


class TrTeData(Dataset):
    """Concatenate Data wrapper, concatenate train and test"""
    allowed_input_channels = 3
    def __init__(self, dataset_name, transform=None):
        dataset_name = dataset_name_tr(dataset_name)
        tv_method = getattr(torchvision.datasets, dataset_name)
        trainset_temp = tv_method(root='./data', train=True, download=True, transform=transform)
        testset_temp = tv_method(root='./data', train=False, download=False, transform=transform)
        trte = ConcatDataset((trainset_temp, testset_temp))
        self.data = trte
        self.num_classes = len(trainset_temp.classes)
        self.num_channels = trainset_temp.data.shape[-1]
        if (self.num_channels > 1) and (self.num_channels != self.allowed_input_channels):
            self.num_channels = 1

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)



class SubDatasetByIndices(Dataset):
    """Used for random cross validation"""
    def __init__(self, dataset_name, indices, transform=None):
        trte_ds = TrTeData(dataset_name, transform)
        self.num_classes = trte_ds.num_classes
        self.num_channels = trte_ds.num_channels
        self.subset = torch.utils.data.Subset(trte_ds, indices)
    def __len__(self):
        return self.subset.__len__()

    def __getitem__(self, idx):
        return self.subset.__getitem__(idx)


class SubdomainDataset(Dataset):
    """Used for VGMM cross validation, Torch Dataset gathering data from input subdomain indice, is an index set"""
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
            _, self.global_index = InputDataset.concatenate_data_from_dir(config_volatile)
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
        trte = TrTeData(config_volatile.dataset_name, transform)
        self.subset = torch.utils.data.Subset(trte.data, self.all_inds)
        self.num_classes = trte.num_classes
        self.num_channels = trte.num_channels


    def __len__(self):
        return self.subset.__len__()

    def __getitem__(self, idx):
        return self.subset.__getitem__(idx)
