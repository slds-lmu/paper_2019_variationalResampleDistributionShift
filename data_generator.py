import numpy as np
# from utils import *
import utils as utils_parent
from VGMM import VGMM
from visualization import T_SNE_Plot
import json
import config

def split_data_according_to_label(z,y,num_labels):
    d = {}
    for i in range(num_labels):
        # d[i]: represent the index of data with label i
        d[str(i)] = np.where(y[:,i]==1)[0]
    return d


def concatenate_index_array(d,num_labels,num_clusters):
    pos = {}
    # pos[ij]: label i, cluster j
    for j in range(num_clusters):
        # init with data of label 0 in cluster j
        p = d[str(0) + str(j)]
        # concatenate p with label i in cluster j
        for i in np.arange(1, num_labels):
            p = np.concatenate((p, d[str(i) + str(j)]), axis=None)
        pos[str(j)] = p.tolist()

    return pos # pos[j]: represent cluster j with multi-label




def concatenate_data_from_dir(data_path,num_labels,num_clusters):
    pos ={}
    # pos[j]:cluster j
    global_index = {}

    for i in range(num_labels):
        path = data_path + "/L" + str(i)
        # z = np.load(path + "/z.npy")
        z = np.load(path + config.z_name)
        # y is the index dictionary with respect to global data
        # y = np.load(path + "/y.npy")
        y = np.load(path + config.y_name)
        # cluster_predict = np.load(path + "/cluster_predict.npy")
        cluster_predict = np.load(path + config.cluster_predict_npy_name)
        if i == 0:
            for j in range(num_clusters):
                pos[str(j)] = z[np.where(cluster_predict == j)]
                global_index[str(j)] = y[np.where(cluster_predict==j)]
        else:
            for j in range(num_clusters):
                pos[str(j)] = np.concatenate((pos[str(j)],z[np.where(cluster_predict == j)]))
                global_index[str(j)] = np.concatenate((global_index[str(j)],y[np.where(cluster_predict==j)]))
    return pos,global_index


def generate_metadata(m,dict,num_clusters):
    for i in range(num_clusters):
        d = dict[str(i)]
        m[d] = i
    return m


def cluster_for_each_label(data_path,num_labels,num_clusters):
    # z = np.load(data_path+"/z.npy")
    z = np.load(data_path + config.z_name)
    # global ground truth
    # y = np.load(data_path+"/y.npy")[:z.shape[0]]
    y = np.load(data_path + config.y_name)[:z.shape[0]]
    d_label = split_data_according_to_label(z,y,num_labels)
    # cluster data of each label
    vgmm = VGMM()
    pos = {}
    for i in range(num_labels):
        print("cluster label "+str(i))
        # extract data of label i
        data = z[d_label[str(i)]]
        # extract global index of data with label i
        data_pos = d_label[str(i)]
        _,data_pred = vgmm.cluster(data)
        for j in range(num_clusters):
            # store the index of cluster j into dictionary ij, i represent label i , cluster j
            pos[str(i) + str(j)] = data_pos[np.where(data_pred == j)[0]]

    # concatenate index array
    pos_index_cluster = concatenate_index_array(pos,num_labels=num_labels,num_clusters=num_clusters)
    # vgmm.save_dict(data_path+"/cluster_dict.json",pos_index_cluster)
    vgmm.save_dict(data_path + config.cluster_index_json_name,pos_index_cluster)

    #generate metadata for visualization
    m = np.zeros(y.shape)
    m = generate_metadata(m,pos_index_cluster,num_clusters=num_clusters)
    # vgmm.save_predict(data_path+"/cluster_predict.tsv",m)
    vgmm.save_predict(data_path + config.cluster_predict_tsv_name,m)
    print(z.shape)
    T_SNE_Plot(z,pos_index_cluster,num_clusters,data_path)

def global_cluster(result_path,z):
    # cluster latent space using VGMM
    print("cluster-vgmm")
    vgmm = VGMM()
    dict, X_prediction_vgmm = vgmm.cluster(z)

    # save the result of clustering
    # path = result_path + "/" + "cluster_dict.json"
    path = result_path + config.cluster_index_json_name
    vgmm.save_dict(path, dict)
    # path = result_path + "/" + "cluster_predict.tsv"
    path = result_path + config.cluster_predict_tsv_name
    vgmm.save_predict(path, X_prediction_vgmm)
    # path = result_path + "/" + "cluster_predict.npy"
    path = result_path + config.cluster_predict_npy_name
    np.save(path,X_prediction_vgmm)


def main():
    print("data_generator")
    # load embedded data
    # data_path = "/Users/wangyu/Documents/LMU/Fashion_mnist/mycode/results/VAE_fashion-mnist_64_10"
    # # global transformed latent variable
    # z = np.load(data_path+"/z.npy")
    # # global ground truth
    # y = np.load(data_path+"/y.npy")[:z.shape[0]]
    # # dictionary of index split according to label
    # d_label = split_data_according_to_label(z,y,10)
    #
    # # cluster data of each label
    # vgmm = VGMM()
    # pos = {}
    # for i in range(10):
    #     # extract data of label i
    #     data = z[d_label[str(i)]]
    #     # extract global index of data with label i
    #     data_pos = d_label[str(i)]
    #     _,data_pred = vgmm.cluster(data)
    #     for j in range(5):
    #         # store the index of cluster j into dictionary ij, i represent label i , cluster j
    #         pos[str(i) + str(j)] = data_pos[np.where(data_pred == j)[0]]
    #
    # # concatenate index array
    # # pos_index_cluster[i]: index of data which belongs to cluster i
    # pos_index_cluster = concatenate_index_array(pos,num_labels=10,num_clusters=5)
    # vgmm.save_dict(data_path+"/pos_index_cluster.json",pos_index_cluster)
    #
    # #generate metadata for visualization
    # m = np.zeros(y.shape)
    # m = generate_metadata(m,pos_index_cluster)
    # vgmm.save_predict(data_path+"/pos_index_cluster_predict.tsv",m)

    # T_SNE plot the result of clustering
    # T_SNE_Plot(z,pos_index_cluster)
if __name__ == '__main__':
    main()

