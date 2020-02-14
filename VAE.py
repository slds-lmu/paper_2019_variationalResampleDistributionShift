#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from ops import *

# from this project
import utils_parent
import mdataset_class
import prior_factory as prior
from data_manipulator import split_data_according_to_label
epsilon4stddev = 1e-6

class VAE(object):
    model_name = "VAE"     # name for checkpoint

    def __init__(self, sess, label, config_manager):
        self.sess = sess
        self.config_manager = config_manager
        #
        self.dataset_name = config_manager.dataset_name
        self.checkpoint_dir = config_manager.checkpoint_dir
        self.result_dir = config_manager.result_dir
        self.log_dir = config_manager.log_dir
        self.epoch = config_manager.epoch
        self.batch_size = config_manager.batch_size
        self.num_labels = config_manager.num_labels
        self.z_dim = config_manager.z_dim
        #
        self.label = label
        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # test
        self.sample_num = 64  # number of generated images to be saved
        ds = mdataset_class.InputDataset(self.dataset_name, label, self.num_labels)
        self.data_X = ds.data_X
        self.data_y = ds.data_y
        # get number of batches for a single epoch
        self.num_batches = len(self.data_X) // self.batch_size

        # tensor order and dim
        self.input_height = self.data_X.shape[1]
        self.input_width = self.data_X.shape[2]
        self.output_height = self.input_height
        self.output_width = self.input_width

        self.c_dim = self.data_X.shape[3]

    # Gaussian Encoder
    def encoder(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            # bn means  batch normalization
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = epsilon4stddev + tf.nn.softplus(gaussian_params[:, self.z_dim:])

        return mean, stddev

    # Bernoulli decoder
    def decoder(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * int(self.input_height/4) * int(self.input_height/4), scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            net = tf.reshape(net, [self.batch_size, int(self.input_height/4), int(self.input_height/4), 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, int(self.input_height/2), int(self.input_width/2), 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
                   scope='de_bn3'))
            # kernel height 4, kernel width 4, stride height 4, stride width 4
            # value, filter, output_shape, strides,
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.input_height, self.input_width, 1], 4, 4, 2, 2, name='de_dc4'))  # heigh is shape[1] so comes first
            # Computes sigmoid of x element-wise
            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        # [bs, height, width, channel]

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        # [bs, z_dim]

        """ Loss Function """
        # encoding
        self.mu, sigma = self.encoder(self.inputs, is_training=True, reuse=False)

        # sampling by re-parameterization technique
        z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        out = self.decoder(z, is_training=True, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - self.KL_divergence

        self.loss = -ELBO

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.neg_loglikelihood, self.KL_divergence],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, nll_loss, kl_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    utils_parent.save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + utils_parent.check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = prior.gaussian(self.batch_size, self.z_dim)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        utils_parent.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                                 utils_parent.check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ learned manifold """
        if self.z_dim == 2:
            assert self.z_dim == 2

            z_tot = None
            id_tot = None
            for idx in range(0, 100):
                #randomly sampling
                id = np.random.randint(0,self.num_batches)
                batch_images = self.data_X[id * self.batch_size:(id + 1) * self.batch_size]
                batch_labels = self.data_y[id * self.batch_size:(id + 1) * self.batch_size]

                z = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})

                if idx == 0:
                    z_tot = z
                    id_tot = batch_labels
                else:
                    z_tot = np.concatenate((z_tot, z), axis=0)
                    id_tot = np.concatenate((id_tot, batch_labels), axis=0)

            utils_parent.save_scattered_image(z_tot, id_tot, -4, 4, name=utils_parent.check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_learned_manifold.png')

    @property
    def model_dir(self):
        return self.config_manager.get_model_dir(self.label)
        # return "{}_{}_{}_{}/{}".format(
        #     self.model_name, self.dataset_name,
        #     self.batch_size, self.z_dim, "L"+str(self.label))

    def super_model_dir(self):
        return self.config_manager.get_super_model_dir()
        # return "{}_{}_{}_{}".format(
        #     self.model_name, self.dataset_name,
        #     self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def transform(self):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        # get batch data
        # return r are only (69952,62) from original dataset which are 70000, 50 datapoints
        #
        # Reload the data without shuffling before mapping it to z space
        noshuffle_data_X = self.data_X
        noshuffle_data_y = self.data_y
        batch_images = noshuffle_data_X[0:self.batch_size]
        r = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})
        for idx in range(1, self.num_batches):     # from the beginning to the end of the dataset, each time do batchsize, conform to the net tensor definition
            batch_images = noshuffle_data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
            z = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})
            r = tf.concat([r, z], 0)
        return r, noshuffle_data_y
