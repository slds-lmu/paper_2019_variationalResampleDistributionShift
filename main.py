#import os
## standard
import argparse
import numpy as np
import tensorflow as tf

## from this project
import utils_parent as utils_parent
import data_generator as data_generator
from VAE import VAE
from VGMM import VGMM
from ACGAN import ACGAN
from data_generator import *
from visualization import *
from config_manager import config_manager

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of embedding"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'fashion-mnist', 'cifar10', 'celebA'],
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
    parser.add_argument('--labeled',type=bool,default=False,help="whether train on specific labeled data")
    parser.add_argument('--cluster',type=bool,default=False,help="whether cluster using latent space")
    parser.add_argument('--num_labels',type=int,default=10,help="number of labels")
    parser.add_argument('--model_name',type=str,default='VAE',help="the name of model to be trained")
    parser.add_argument('--plot',type=bool,default=True,help="visualise the result of cluster")
    parser.add_argument('--num_clusters',type=int,default=5,help="number of clusters")
    return check_args(parser.parse_args())

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

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    config_m = config_manager(args)

    if args.model_name == "VAE":   # choose between VAE or ACGAN for the latent mapping
        if args.labeled:
            # train model on data splited according to label, alternative is to train VAE on all classes to get a common latent representation for visualization and calculation of wasserstein distance
            # declare global z and index dictionary to store the result of resampling

            # declare instance for VAE for each label
            for i in range(args.num_labels):
                # reset the graph
                tf.reset_default_graph()
                # open session
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    vae = None
                    vae = VAE(sess,
                              epoch=args.epoch,
                              batch_size=args.batch_size,
                              z_dim=args.z_dim,
                              dataset_name=args.dataset,
                              checkpoint_dir=args.checkpoint_dir,
                              result_dir=args.result_dir,
                              log_dir=args.log_dir,
                              label=i,
                              config_manager=config_m
                              )
                    # build graph
                    vae.build_model()

                    # show network architecture
                    utils_parent.show_all_variables()

                    # launch the graph in a session
                    vae.train()
                    print(" [*] Training finished!")

                    # visualize learned generator
                    vae.visualize_results(args.epoch - 1)
                    print(" [*] Testing finished!")

                    # save the transformed latent space into result dir
                    # filepath = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                    z, noshuffle_data_y = vae.transform()
                    z = z.eval()
                    filepath = config_m.get_data_path_for_each_label(i) + config_m.z_name
                    if not tf.gfile.Exists(filepath):
                        np.save(filepath, z)
                        # filepath = args.result_dir + "/" + vae.model_dir + "/" + "y.npy"
                        np.save(config_m.get_data_path_for_each_label(i)+ config_m.y_name, noshuffle_data_y)




                    if args.cluster:
                        # cluster latent space using VGMM
                        # result_path = args.result_dir + "/" + vae.model_dir
                        # cluster the transformed latent space, and store the dictionary and prediction into result_path

                        global_cluster(config_m.get_data_path_for_each_label(i),z)

            print(" [*] Training and Testing for all label finished!")
            # concatenate clustered data into one dict after clustering
            # result_path = args.result_dir + "/" + vae.super_model_dir()
            data_dict,global_index = concatenate_data_from_dir(data_path=config_m.get_data_path(),num_labels=config_m.num_labels,num_clusters=config_m.num_clusters)
            # save global index for cluster data
            # np.save(config_m.get_data_path()+"/global_index_cluster_data.npy",global_index,allow_pickle=True)
            np.save(config_m.get_data_path()+ config_m.global_index_name, global_index,allow_pickle=True)
            T_SNE_Plot_with_datadict(data_dict=data_dict,num_clusters=config_m.num_clusters,result_path=config_m.get_data_path())
            # write_path_to_config(config_m.get_data_path())
            config_m.write_config_file()
        else:   # without label, built up a common latent representation of all instances from all classes
            # declare instance for VAE
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                vae = None
                vae = VAE(sess,
                          epoch=args.epoch,
                          batch_size=args.batch_size,
                          z_dim=args.z_dim,
                          dataset_name=args.dataset,
                          checkpoint_dir=args.checkpoint_dir,
                          result_dir=args.result_dir,
                          log_dir=args.log_dir,
                          config_manager=config_m
                          )
                # build graph
                vae.build_model()

                # show network architecture
                utils_parent.show_all_variables()

                # launch the graph in a session
                vae.train()
                print(" [*] Training finished!")

                # visualize learned generator
                vae.visualize_results(args.epoch - 1)
                print(" [*] Testing finished!")

                # save the transformed latent space into result dir
                # filepath = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                # filepath = config_m.get_data_path_for_each_label(-1)+"/z.npy"
                filepath = config_m.get_data_path_for_each_label(-1)+ config_m.z_name
                if not tf.gfile.Exists(filepath):
                    z, noshuffle_data_y = vae.transform()
                    z = z.eval()
                    np.save(filepath, z)
                    np.save(config_m.get_data_path_for_each_label(-1)+ config_m.y_name, noshuffle_data_y)


                if args.cluster:
                    # result_path = args.result_dir + "/" + vae.model_dir
                    # filepath = config_m.get_data_path_for_each_label(-1) + "/cluster_dict.json"
                    filepath = config_m.get_data_path_for_each_label(-1) + config_m.cluster_index_json_name
                    if not tf.gfile.Exists(filepath):
                        data_generator.cluster_for_each_label(config_m.get_data_path_for_each_label(-1),num_labels=config_m.num_labels,num_clusters=config_m.num_clusters)
                # result_path = args.result_dir + "/" + vae.super_model_dir()
                # write_path_to_config(config_m.get_data_path())
                config_m.write_config_file()
    elif args.model_name =="ACGAN":
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
    main()
