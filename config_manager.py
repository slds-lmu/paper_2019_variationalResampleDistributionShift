import argparse
import get_source_code_dir
import utils_parent

class ConfigManager(object):
    global_index_name = "/global_index_cluster_data.npy"
    TSNE_data_name = "/TSNE_transformed_data_dict.npy"
    cluster_index_json_name = "/cluster_dict.json"
    cluster_predict_tsv_name = "/cluster_predict.tsv"
    cluster_predict_npy_name = "/cluster_predict.npy"
    z_name ="/z.npy"
    y_name = "/y.npy"
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.labeled = args.labeled
        self.cluster = args.cluster
        self.model_name = args.model_name
        self.num_labels = args.num_labels
        self.num_clusters = args.num_clusters
        self.plot = args.plot
        #
        self.rst_dir = get_source_code_dir.current_dir
        #
        self.log_dir = "./config.py"  # the volatile file should live in the directory where the program is exceculted
        # hard coded
        self.global_index_name = ConfigManager.global_index_name
        self.TSNE_data_name = ConfigManager.TSNE_data_name
        self.cluster_index_json_name = ConfigManager.cluster_index_json_name
        self.cluster_predict_tsv_name = ConfigManager.cluster_predict_tsv_name
        self.cluster_predict_npy_name = ConfigManager.cluster_predict_npy_name
        self.z_name =ConfigManager.z_name
        self.y_name = ConfigManager.y_name


    def get_model_dir(self,label):
        return "{}_{}_{}_{}/{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim, "L" + str(label))


    def get_super_model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def get_data_path(self):
        return "{}/{}_{}_{}_{}".format(
            self.result_dir,self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def get_result_path(self):
        return "{}/{}_{}_{}_{}".format(
            self.result_dir,self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def get_data_path_for_each_label(self,label):
        # generate path to store the production while training according to label
        return "{}/{}_{}_{}_{}/{}".format(
            self.result_dir,self.model_name, self.dataset_name,
            self.batch_size, self.z_dim, "L" + str(label))

    def write_config_file(self):
        file = open(self.log_dir, 'w')
        file.write("rst_dir = '{}'".format(self.rst_dir))
        file.write('\n')
        file.write("data_path = '{}'".format(self.get_data_path()))
        file.write('\n')
        file.write("result_path = '{}'".format(self.get_result_path()))
        file.write('\n')
        file.write('statistic_name4d_t = "/L-1/TSNE_transformed_data_dict.npy"')
        file.write('\n')
        file.write('statistic_name4d_s = "/TSNE_transformed_data_dict.npy"')
        file.write('\n')
        file.write('num_clusters={}'.format(self.num_clusters))
        file.write('\n')
        file.write('num_labels={}'.format(self.num_labels))
        file.write('\n')
        file.write("dataset_name='{}'".format(self.dataset_name))
        file.write('\n')
        file.write("global_index_name='{}'".format(self.global_index_name))
        file.write('\n')
        file.write("TSNE_data_name='{}'".format(self.TSNE_data_name))
        file.write('\n')
        file.write("cluster_index_json_name='{}'".format(self.cluster_index_json_name))
        file.write('\n')
        file.write("cluster_predict_tsv_name='{}'".format(self.cluster_predict_tsv_name))
        file.write('\n')
        file.write("cluster_predict_npy_name='{}'".format(self.cluster_predict_npy_name))
        file.write('\n')
        file.write("z_name='{}'".format(self.z_name))
        file.write('\n')
        file.write("y_name='{}'".format(self.y_name))
        file.write('\n')
        file.close()

## parsing and configuration
def parse_args(raw_args=None):
    desc = "Tensorflow implementation of embedding"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rst_dir', type=str, default='./', help='absolute result directory, default to be the same folder as where the code lies')
    parser.add_argument('--dataset', type=str, default='fashion-mnist', choices=['fashion-mnist', 'cifar10'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    # arguments specified for model
    parser.add_argument('--labeled', type=bool, default=False, help="whether train on specific labeled data")
    parser.add_argument('--cluster', type=bool, default=False, help="whether cluster using latent space")
    parser.add_argument('--num_labels', type=int, default=10, help="number of labels")
    parser.add_argument('--model_name', type=str, default='VAE', help="the name of model to be trained")
    parser.add_argument('--plot', type=bool, default=True, help="visualise the result of cluster")
    parser.add_argument('--num_clusters', type=int, default=5, help="number of clusters")
    return check_args(parser.parse_args(raw_args))

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    utils_parent.check_folder(args.checkpoint_dir)

    # --result_dir
    utils_parent.check_folder(args.result_dir)

    # --log_dir
    utils_parent.check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    # --num_labels
    assert args.num_labels >=1, 'number of labels must be larger than or equal to one'

    return args


