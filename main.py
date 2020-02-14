## standard import
import numpy as np
import tensorflow as tf

## from this project
import utils_parent
import data_manipulator
import visualization
from VAE import VAE
from ACGAN import ACGAN
from config_manager import ConfigManager
from config_manager import parse_args

def embed_cluster(raw_args=None):
    # parse arguments
    args = parse_args(raw_args)
    if args is None:
      exit()

    if args.dataset == 'fashion-mnist': args.dataset = "FashionMNIST"
    elif args.dataset =='cifar10': args.dataset = "CIFAR10"

    config_m = ConfigManager(args)

    if args.model_name == "VAE":   # choose between VAE or ACGAN for the latent mapping
        if not args.labeled: i = -1 # train VAE on all classes to get a common latent representation for visualization and calculation of wasserstein distance
        else: i = 0 # train model on data splited according to label, starting with index 0, declare instance for VAE for each label
        while i < (args.num_labels):
            # reset the graph
            tf.reset_default_graph()
            # open session
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                vae = None
                vae = VAE(sess,
                          label=i,  # the difference with respect to non-label sensitive, by default label = -1, without label, built up a common latent representation of all instances from all classes
                          config_manager=config_m
                          )
                # build graph
                vae.build_model()

                # show network architecture
                if i < 1: # only once
                    utils_parent.show_all_variables()
                # launch the graph in a session
                vae.train()
                print(" [*] Training finished!")

                # visualize learned generator
                vae.visualize_results(args.epoch - 1)
                print(" [*] Fake Image saved!")

                # save the transformed latent space into result dir
                filepath = config_m.get_data_path_for_each_label(i) + config_m.z_name

                if (not tf.gfile.Exists(filepath)) or args.labeled:
                    z, noshuffle_data_y = vae.transform()
                    z = z.eval()  # z is not used for cluster in the case of args.labeled=False, but should be used to calculate wasserstein distance, calculating z for all data can be very time consuming!
                    print(" [*] latent representation calculation finished!")

                if not tf.gfile.Exists(filepath):
                    np.save(filepath, z)
                    np.save(config_m.get_data_path_for_each_label(i)+ config_m.y_name, noshuffle_data_y)
                    print(" [*] latent representation and correponding class label saved!")
                print(" [*] going to enter clustering (or not) ....")
                if args.cluster:
                    print(" [*] clustering....")
                    # cluster latent space using VGMM: cluster the transformed latent space, and store the dictionary and prediction into result_path
                    if not args.labeled:   # if not divided by label, cluster by label and merge
                        print(" [*] VAE training without label, run cluster for each label now")
                        filepath = config_m.get_data_path_for_each_label(-1) + config_m.cluster_index_json_name  # "/cluster_dict.json"
                        if not tf.gfile.Exists(filepath):
                            data_manipulator.cluster_common_embeding_labelwise(config_m.get_data_path_for_each_label(-1), num_labels=config_m.num_labels, num_clusters=config_m.num_clusters)
                        print(" [*] after VAE training without label information, cluster by each label and merge finished and saved!")
                    else:  # data comes in, divided by label
                        data_manipulator.cluster_save2disk_label(config_m.get_data_path_for_each_label(i), z, config_m.num_clusters)  # if data already comes by label, then run VGMM directly
                        print("vgmm cluster on label", i, "finished")
            if not args.labeled:
                i = args.num_labels  # while
                print("label", i, "finished")
            i = i+1 # while
        # After for loop is finished
        if args.labeled:  # merge the clusters from each label
            print(" [*] merging clusters from each label....")
            # concatenate clustered data into one dict after clustering
            data_dict, global_index = data_manipulator.concatenate_data_from_dir(config_m)
            # global_index is the final result of this routine
            # save global index for cluster data
            np.save(config_m.get_data_path()+ config_m.global_index_name, global_index, allow_pickle=True)
            visualization.T_SNE_Plot_with_datadict(data_dict=data_dict, num_clusters=config_m.num_clusters, result_path=config_m.get_data_path())
        config_m.write_config_file()
        print("* volatile configuration file written")
    # not yet used
    elif args.model_name == "ACGAN":
        # declare instance for ACGANG
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            acgan = None
            acgan = ACGAN(sess,
                      epoch=args.epoch,
                      batch_size=args.batch_size,
                      z_dim=args.z_dim,
                      dataset_name=args.dataset,
                      checkpoint_dir=args.checkpoint_dir,
                      result_dir=args.result_dir,
                      log_dir=args.log_dir)
            # build graph
            acgan.build_model()

            # show network architecture
            utils_parent.show_all_variables()

            # launch the graph in a session
            acgan.train()
            print(" [*] Training finished!")

            # visualize learned generator
            acgan.visualize_results(args.epoch - 1)
            print(" [*] Testing finished!")

    else:
        raise NotImplementedError


if __name__ == '__main__':
    embed_cluster()
