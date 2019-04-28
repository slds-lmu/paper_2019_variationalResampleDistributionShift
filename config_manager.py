class config_manager(object):
    def __init__(self,args):
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
        file = open('config.py', 'w')
        file.write("SOURCE_URL_Mnist = 'http://yann.lecun.com/exdb/mnist/'")
        file.write('\n')
        file.write("SOURCE_URL_FMnist = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'")
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
        file.write('dataset_name={}'.format(self.dataset_name))
        file.write('\n')
        file.close()