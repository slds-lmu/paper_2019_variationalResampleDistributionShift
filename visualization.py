## Importing required Libraries
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import check_folder
import json
# import matplotlib as plt
import matplotlib.pyplot as plt
from utils import *
import config

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
        plt.savefig(result_path + "/TSNE" + str(i) + ".png")
    plt.grid(True)
    plt.savefig(result_path+"/TSNE.jpg")
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
        plt.savefig(result_path + "/TSNE"+str(i)+".png")
    plt.grid(True)
    plt.savefig(result_path+"/TSNE.jpg")
    # np.save(result_path + "/TSNE_transformed_data_dict.npy", fashion_pca_tsne)
    np.save(result_path + config.TSNE_data_name, fashion_pca_tsne)



def visualization(log_path,data_path):
    ## Get working directory
    PATH = os.getcwd()

    ## Path to save the embedding and checkpoints generated
    # LOG_DIR = PATH + '/project-tensorboard/log-1/'
    LOG_DIR = PATH + log_path
    check_folder(LOG_DIR)

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

if __name__ == '__main__':
    print("visualization")
    # log_path = "/project-tensorboard/VAE_10"
    # data_path = "/Users/wangyu/Documents/LMU/Fashion_mnist/mycode/results/VAE_fashion-mnist_64_10"
    # visualization(log_path,data_path)