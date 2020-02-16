import torch
from mdataset_class import SubdomainDataset, SubDatasetByIndices
def split_te_tr_val(config_volatile, method, train_eval_list, test_list, transform_train, transform_test, debug = False):
    debug_frac = 0.01
    train_val_frac = 0.8
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

