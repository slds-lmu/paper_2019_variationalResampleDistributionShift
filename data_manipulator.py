import numpy as np
import utils_parent as utils_parent
from VGMM import VGMM
from visualization import T_SNE_Plot
from config_manager import ConfigManager
from mdataset_class import InputDataset
import json
import os

def cluster_save2disk_label(result_path, phi, num_clusters):
    print("clustering given variational parameter phi(mu/sigma) data from arguments")
    vgmm = VGMM(num_clusters)
    mdict, X_prediction_vgmm = vgmm.cluster(phi)
    # save the result of clustering
    path = result_path + ConfigManager.cluster_index_json_name  # path = result_path + "/" + "cluster_dict.json"
    vgmm.save_dict(path, mdict)
    path = result_path + ConfigManager.cluster_predict_tsv_name # path = result_path + "/" + "cluster_predict.tsv"
    vgmm.save_predict(path, X_prediction_vgmm)
    path = result_path + ConfigManager.cluster_predict_npy_name # path = result_path + "/" + "cluster_predict.npy"
    np.save(path, X_prediction_vgmm)
    print("cluster results saved to labeled path")




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

def cluster_common_embeding_labelwise(y, data_path, num_labels, num_clusters):
    z = np.load(data_path + ConfigManager.z_name)
    # global ground truth
    # y = np.load(data_path + ConfigManager.y_name)[:z.shape[0]]
    d_label = InputDataset.split_data_according_to_label(y, num_labels)
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
