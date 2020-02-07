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

