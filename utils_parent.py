"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import matplotlib
from six.moves import xrange
import matplotlib.pyplot as plt
import os, gzip
import json

import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import urllib
import config

import torch
from torch.utils.data.dataset import ConcatDataset
import torchvision

def load_torchvision_data2np(dataset_name = "CIFAR10", num_classes = 10, shuffle=True, seed=547, allowed_input_channels = [1, 3]):

    tv_method = getattr(torchvision.datasets, dataset_name)
    # train = True
    # function transform is defined in this module as a hook to torchvision
    trainset_temp = tv_method(root='./data', train=True, download=True, transform=transform)
    trX = trainset_temp.data
    if(trX.shape[-1] != allowed_input_channels[0] and trX.shape[-1] != allowed_input_channels[1]): trX = trX.unsqueeze(-1)
    trY = trainset_temp.targets
    # train = False
    testset_temp = tv_method(root='./data', train=False, download=False, transform=transform)
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




def load_mnist(dataset_name, shuffle=True, seed=547):
    trainset_temp = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trX = trainset_temp.data
    trX = trX.reshape((60000, 28, 28, 1))
    trY = trainset_temp.targets
    testset_temp = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    teX = testset_temp.data
    teX = teX.reshape((10000, 28, 28, 1))
    teY = testset_temp.targets
    cd = ConcatDataset((trainset_temp, testset_temp))
    #return cd.data, cd.targets
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    yy = np.zeros((len(y), 10))
    yy[np.arange(len(y)), y] = 1
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(yy)
    return X/255., yy


# Download MNIST data if there is no data in dir
# borrowed from https://github.com/hwalsuklee/tensorflow-mnist-VAE.git
def maybe_download(SOURCE_URL,DATA_DIRECTORY,filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

# Mnist
def load_mnist_old(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    if dataset_name =="mnist":
        SOURCE_URL = config.SOURCE_URL_Mnist
    elif dataset_name =="fashion-mnist":
        SOURCE_URL = config.SOURCE_URL_FMnist
    else:
        raise Exception('The value of dataset_name could only be: {}'.format("mnist or fashion-mnist"))
    # Get the data.
    maybe_download(SOURCE_URL,data_dir,'train-images-idx3-ubyte.gz')
    maybe_download(SOURCE_URL,data_dir,'train-labels-idx1-ubyte.gz')
    maybe_download(SOURCE_URL,data_dir,'t10k-images-idx3-ubyte.gz')
    maybe_download(SOURCE_URL,data_dir,'t10k-labels-idx1-ubyte.gz')

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    # one-hot code
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# transfer .npy file to .csv file
def npy2csv(fileinput, fileoutput):
    data = np.load(fileinput)
    np.savetxt(fileoutput, data, delimiter=",")

#split data accordingt to label
def prepare_data():
    return 0

def save_dict(path,dict):
    with open(path, 'w') as f:
        json.dump(dict, f)


def load_dict(path):
    with open(path) as f:
        my_dict = json.load(f)
    return my_dict

def write_path_to_config(result_path):
    file = open('config.py', 'w')
    file.write("SOURCE_URL_Mnist = 'http://yann.lecun.com/exdb/mnist/'")
    file.write('\n')
    file.write("SOURCE_URL_FMnist = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'")
    file.write('\n')
    file.write("data_path = '{}'".format(result_path))
    file.write('\n')
    file.write("result_path = '{}'".format(result_path))
    file.write('\n')
    file.write('statistic_name4d_t = "/L-1/TSNE_transformed_data_dict.npy"')
    file.write('\n')
    file.write('statistic_name4d_s = "/TSNE_transformed_data_dict.npy"')
    file.write('\n')
    file.close()

# write results dict like: {‘0’:{‘test’:[{‘epoch’:1,‘loss’:1},{‘epoch’:2,‘loss’:2}],‘train’:[{‘epoch’:1,‘loss’:1},{‘epoch’:2,‘loss’:2}]}}
def write_results_to_csv(path,dict):
    import csv
    # with open('results.csv', 'w', newline='') as csvfile:
    with open(path, 'w', newline='') as csvfile:
        for key, val in dict.items():
            w = csv.writer(csvfile)
            w.writerow({"cv:", key})
            for key, val in val.items():
                # val:[{'epoch':1,'loss':1},{'epoch':2,'loss':2}]
                w.writerow({"mode:", key})
                fieldnames = val[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for d in val:
                    writer.writerow(d)

# write results dict like {'0':{'test':{'acc':1,'loss':1},'train':{'acc':1,'loss':1}}, '1':{'test':{'acc':1,'loss':1},'train':{'acc':1,'loss':1}}}
def write_results_convnet_to_csv(path,dict):
    import csv
    with open(path, 'w', newline='') as csvfile:
        head = True
        for key, val in dict.items():
            w = csv.writer(csvfile)
            for key, val in val.items():
                fieldnames = val.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if head:
                    writer.writeheader()
                    head = False
                writer.writerow(val)
