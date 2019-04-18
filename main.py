import os

## VAE Variants
from VAE import VAE
from utils import show_all_variables
from utils import check_folder
import tensorflow as tf
import argparse
from VGMM import VGMM
from ACGAN import ACGAN
import numpy as np
from data_generator import *
from visualization import *
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='fashion-mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
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
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --log_dir
    check_folder(args.log_dir)

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

    if args.model_name == "VAE":
        if args.labeled:
            # train model on data splited according to label
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
                              label=i
                              )
                    # build graph
                    vae.build_model()

                    # show network architecture
                    show_all_variables()

                    # launch the graph in a session
                    vae.train()
                    print(" [*] Training finished!")

                    # visualize learned generator
                    vae.visualize_results(args.epoch - 1)
                    print(" [*] Testing finished!")

                    # save the transformed latent space into result dir
                    z = vae.transform()
                    z = z.eval()
                    path = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                    np.save(path, z)
                    path = args.result_dir + "/" + vae.model_dir + "/" + "y.npy"
                    np.save(path, vae.data_y)

                    # filepath = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                    # if not tf.gfile.Exists(filepath):
                    #     z = vae.transform()
                    #     z = z.eval()
                    #     path = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                    #     np.save(path, z)
                    #     path = args.result_dir + "/" + vae.model_dir + "/" + "y.npy"
                    #     np.save(path, vae.data_y)



                    if args.cluster:
                        # cluster latent space using VGMM
                        result_path = args.result_dir + "/" + vae.model_dir
                        # cluster the transformed latent space, and store the dictionary and prediction into result_path
                        global_cluster(result_path,z)

            print(" [*] Training and Testing for all label finished!")
            # concatenate clustered data into one dict after clustering
            result_path = args.result_dir + "/" + vae.super_model_dir()
            data_dict = concatenate_data_from_dir(result_path,10,5)
            T_SNE_Plot_with_datadict(data_dict,5,result_path)


        else:
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
                          log_dir=args.log_dir)
                # build graph
                vae.build_model()

                # show network architecture
                show_all_variables()

                # launch the graph in a session
                vae.train()
                print(" [*] Training finished!")

                # visualize learned generator
                vae.visualize_results(args.epoch - 1)
                print(" [*] Testing finished!")
                # save the transformed latent space into result dir
                filepath = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                if not tf.gfile.Exists(filepath):
                    z = vae.transform()
                    z = z.eval()
                    path = args.result_dir + "/" + vae.model_dir + "/" + "z.npy"
                    np.save(path, z)
                    path = args.result_dir + "/" + vae.model_dir + "/" + "y.npy"
                    np.save(path, vae.data_y)


                if args.cluster:
                    result_path = args.result_dir + "/" + vae.model_dir
                    filepath = result_path + "/cluster_dict.json"
                    cluster_for_each_label(result_path,10,5)
                    # if not tf.gfile.Exists(filepath):
                    #     cluster_for_each_label(result_path,10,5)

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
            show_all_variables()

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
