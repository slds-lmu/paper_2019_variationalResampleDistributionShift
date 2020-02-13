## Importing required Libraries
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from utils import check_folder
import utils_parent as utils_parent
from config_manager import ConfigManager
import json
# import matplotlib as plt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from utils import *
# import data_manipulator
# from data_manipulator import concatenate_data_from_dir

def T_SNE_Plot(data_x,pos,num_clusters,result_path):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    fashion_pca_tsne = {}
    color_dict = {'0':'r','1':'g','2':'plum','3':'b','4':'k','5':'y'}

    for i in range(num_clusters):
        X = data_x[pos[str(i)]]
        print(X.shape)
        # y = data_y[pos[str(i)]]
        if X.shape[1]>50:
            pca_50 = PCA(n_components=10)
            pca_result_50 = pca_50.fit_transform(X)
            fashion_pca_tsne[str(i)] = TSNE().fit_transform(pca_result_50)
        else:
            fashion_pca_tsne[str(i)] = TSNE().fit_transform(X)
        plt.scatter(fashion_pca_tsne[str(i)][:, 0], fashion_pca_tsne[str(i)][:, 1], color=color_dict[str(i)], alpha=0.1)
        plt.savefig(result_path + "/TSNE" + str(i) + ".pdf")
    plt.grid(True)
    plt.savefig(result_path+"/TSNE.pdf")
    # np.save(result_path+"/TSNE_transformed_data_dict.npy",fashion_pca_tsne)
    np.save(result_path + config.TSNE_data_name,fashion_pca_tsne)

def T_SNE_Plot_with_datadict(data_dict,num_clusters,result_path):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    fashion_pca_tsne = {}
    color_dict = {'0': 'r', '1': 'g', '2': 'plum', '3': 'b', '4': 'k', '5': 'y'}
    for i in range(num_clusters):
        X = data_dict[str(i)]
        if X.shape[1] > 50:
            pca_50 = PCA(n_components=10)
            pca_result_50 = pca_50.fit_transform(X)
            fashion_pca_tsne[str(i)] = TSNE().fit_transform(pca_result_50)
        else:
            fashion_pca_tsne[str(i)] = TSNE().fit_transform(X)
        plt.scatter(fashion_pca_tsne[str(i)][:, 0], fashion_pca_tsne[str(i)][:, 1], color=color_dict[str(i)], alpha=0.1)
        plt.savefig(result_path + "/TSNE"+str(i)+".pdf")
    plt.grid(True)
    plt.savefig(result_path+"/TSNE.pdf")
    # np.save(result_path + "/TSNE_transformed_data_dict.npy", fashion_pca_tsne)
    np.save(result_path + ConfigManager.TSNE_data_name, fashion_pca_tsne)



def visualization(log_path,data_path):
    ## Get working directory
    PATH = os.getcwd()

    ## Path to save the embedding and checkpoints generated
    # LOG_DIR = PATH + '/project-tensorboard/log-1/'
    LOG_DIR = PATH + log_path
    utils_parent.check_folder(LOG_DIR)

    ## Load data
    data = np.load(data_path+"/z.npy")

    # Load the metadata file. Metadata consists your labels. This is optional. Metadata helps us visualize(color) different clusters that form t-SNE
    # metadata = os.path.join(data_path,'/pos_index_cluster_predict.tsv')
    metadata = data_path+"/pos_index_cluster_predict.tsv"
    # Generating PCA and
    # pca = PCA(n_components=50,
    #          random_state = 123,
    #          svd_solver = 'auto'
    #          )
    # df_pca = df
    # data_pca = pca.fit_transform(data)
    # df_pca = df_pca.values

    ## TensorFlow Variable from data
    tf_data = tf.Variable(data)
    # tf_data = tf.Variable(data_pca)
    ## Running TensorFlow Session
    with tf.Session() as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name
        # Link this tensor to its metadata(Labels) file
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

def concatenate_data_from_dir_deprecated(data_path,num_labels,num_clusters):
    pos ={}
    # pos[j]:cluster j
    global_index = {}

    for i in range(num_labels):
        path = data_path + "/L" + str(i)
        # z = np.load(path + "/z.npy")
        z = np.load(path + ConfigManager.z_name)
        # y is the index dictionary with respect to global data
        # y = np.load(path + "/y.npy")
        y = np.load(path + ConfigManager.y_name)
        # cluster_predict = np.load(path + "/cluster_predict.npy")
        cluster_predict = np.load(path + ConfigManager.cluster_predict_npy_name)
        if i == 0:
            for j in range(num_clusters):
                pos[str(j)] = z[np.where(cluster_predict == j)]
                global_index[str(j)] = y[np.where(cluster_predict==j)]
        else:
            for j in range(num_clusters):
                pos[str(j)] = np.concatenate((pos[str(j)],z[np.where(cluster_predict == j)]))
                global_index[str(j)] = np.concatenate((global_index[str(j)],y[np.where(cluster_predict==j)]))
    return pos,global_index
if __name__ == '__main__':
    print("visualization")
    # log_path = "/project-tensorboard/VAE_10"
    # data_path = "/Users/wangyu/Documents/LMU/Fashion_mnist/mycode/results/VAE_fashion-mnist_64_10"
    # visualization(log_path,data_path)

    # t-SNE plot for labeled data
    import config
    data_dict, _ = concatenate_data_from_dir(data_path=config.data_path,
                                                        num_labels=config.num_labels,
                                                        num_clusters=config.num_clusters)
    T_SNE_Plot_with_datadict(data_dict=data_dict, num_clusters=config.num_clusters,
                             result_path=config.data_path)



