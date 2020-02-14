import numpy as np
import utils_parent as utils_parent
from VGMM import VGMM
from visualization import T_SNE_Plot
from config_manager import ConfigManager
import json

def cluster_save2disk_label(result_path, z, num_clusters):
    print("cluster given z data from arguments")
    vgmm = VGMM(num_clusters)
    mdict, X_prediction_vgmm = vgmm.cluster(z)
    # save the result of clustering
    path = result_path + ConfigManager.cluster_index_json_name  # path = result_path + "/" + "cluster_dict.json"
    vgmm.save_dict(path, mdict)
    path = result_path + ConfigManager.cluster_predict_tsv_name # path = result_path + "/" + "cluster_predict.tsv"
    vgmm.save_predict(path, X_prediction_vgmm)
    path = result_path + ConfigManager.cluster_predict_npy_name # path = result_path + "/" + "cluster_predict.npy"
    np.save(path, X_prediction_vgmm)
    print("cluster results saved to labeled path")



def concatenate_data_from_dir(config_volatile):
    pos = {}  # pos[i_cluster] correspond to the z value (concatenated) of cluster i_cluster
    global_index = {}  # global_index['cluster_1'] correspond to the global index with respect to the original data of cluster 1
    data_path, num_labels, num_clusters = config_volatile.data_path, config_volatile.num_labels, config_volatile.num_clusters
    for i_label in range(num_labels):
        path = data_path + "/L" + str(i_label)   #FIXME! $"/L"
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


def split_data_according_to_label(z, y, num_labels):
    d = {}
    for i in range(num_labels):
        # d[i]: represent the index of data with label i
        d[str(i)] = np.where(y[:, i] == 1)[0]
    return d

def generate_metadata(m, mdict, num_clusters):
    for i in range(num_clusters):
        d = mdict[str(i)]
        m[d] = i
    return m



def concatenate_index_array(d, num_labels, num_clusters):
    pos = {}
    # pos['ij']: label i, cluster j
    for j in range(num_clusters):
        # init with data of label 0 in cluster j
        p = d[str(0) + str(j)]
        # concatenate p with label i in cluster j
        for i in np.arange(1, num_labels):
            p = np.concatenate((p, d[str(i) + str(j)]), axis=None)
        pos[str(j)] = p.tolist()

    return pos # pos[j]: represent cluster j with multi-label

def cluster_common_embeding_labelwise(data_path, num_labels, num_clusters):
    z = np.load(data_path + ConfigManager.z_name)
    # global ground truth
    y = np.load(data_path + ConfigManager.y_name)[:z.shape[0]]
    d_label = split_data_according_to_label(z, y, num_labels)
    # cluster data of each label
    vgmm = VGMM(num_clusters)
    pos = {}
    for i in range(num_labels):
        print("cluster label "+str(i))
        # extract data of label i
        data = z[d_label[str(i)]]
        # extract global index of data with label i
        data_pos = d_label[str(i)]
        _, data_pred = vgmm.cluster(data)
        for j in range(num_clusters):
            # store the index of cluster j into dictionary ij, i represent label i , cluster j
            pos[str(i) + str(j)] = data_pos[np.where(data_pred == j)[0]]
    pos_index_cluster = concatenate_index_array(pos,num_labels=num_labels, num_clusters=num_clusters)
    vgmm.save_dict(data_path + ConfigManager.cluster_index_json_name, pos_index_cluster)
    #generate metadata for visualization
    m = np.zeros(y.shape)
    m = generate_metadata(m, pos_index_cluster, num_clusters=num_clusters)
    vgmm.save_predict(data_path + ConfigManager.cluster_predict_tsv_name, m)
    print(z.shape)
    T_SNE_Plot(z, pos_index_cluster, num_clusters, data_path)
