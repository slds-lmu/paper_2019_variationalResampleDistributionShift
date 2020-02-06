from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from utils import *
import config
import utils_parent as utils_parent
import argparse
from sklearn.model_selection import KFold
from data_manipulator import concatenate_data_from_dir
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


DATA_DIR = './data/fashion'


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    input_layer = features["x"]
    print(mode)
    # print(input_layer.shape)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # change the classes to one hot encode
        # "classes": tf.argmax(input=logits, axis=1),
        "classes": tf.one_hot(indices=tf.cast(tf.argmax(input=logits, axis=1), tf.int32), depth=10),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = labels
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def parse_args():
    desc = "Tensorflow implementation CNN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='fashion-mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    # arguments specified for model
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

    return args


def train(X,y,val_x,val_y,test_x,test_y,args,i):
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X},
        y=y,
        batch_size=args.batch_size,
        num_epochs=args.epoch,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": val_x},
        y=val_y,
        num_epochs=1,
        shuffle=False)

    # Test the model using test data from other cluster
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x":test_x},
        y = test_y,
        batch_size = args.batch_size,
        num_epochs = 1,
        shuffle = False
    )

    # Create the Estimator
    model_dir = "{}/convnet_{}_{}_{}_{}".format(args.result_dir,args.dataset,args.batch_size,args.epoch,i)
    utils_parent.check_folder(model_dir)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("*********************eval************************")
    print(eval_results)
    test_results = mnist_classifier.evaluate(input_fn = test_input_fn)
    print("*********************test************************")
    print(test_results)
    return eval_results,test_results



def cross_validation(X,y,split_size=5,args=None):
    results = {}
    kf = KFold(n_splits=split_size)
    i = 0
    for train_eval_idx, test_idx in kf.split(X, y):
        print(train_eval_idx.shape)
        print(test_idx.shape)
        x_train_eval = X[train_eval_idx]
        y_train_eval = y[train_eval_idx]
        test_x = X[test_idx]
        test_y = y[test_idx]
        train_x, val_x, train_y, val_y = train_test_split(x_train_eval, y_train_eval, test_size=0.2, random_state=42)
        eval_result,test_result = train(train_x, train_y,val_x,val_y,test_x,test_y,args,i)
        result = {"train":eval_result,"test":test_result}
        results[str(i)] = result
        i = i+1
    return results



def cross_validation_for_clustered_data(X,y,data_path,num_labels,num_cluster,args):
    print("cross validation for clustered data")
    results = {}
    if not tf.gfile.Exists(data_path+"/global_index_cluster_data.npy"):
        _,global_index = concatenate_data_from_dir(data_path,num_labels=num_labels,num_clusters=num_cluster)
    else:global_index = np.load(data_path+"/global_index_cluster_data.npy",allow_pickle=True)
    for i in range(num_cluster):
        index = global_index.item().get(str(i))
        test_x = X[index]
        test_y = y[index]
        mask = np.ones((y.shape[0],),bool)
        mask[index]=False
        x_train_eval = X[mask]
        y_train_eval = y[mask]
        train_x, val_x, train_y, val_y = train_test_split(x_train_eval, y_train_eval, test_size = 0.2, random_state = 42)
        args.result_dir = "results/clustercnn"
        eval_result,test_result = train(train_x, train_y, val_x, val_y,test_x,test_y, args, i)
        result = {"train":eval_result,"test":test_result}
        results[str(i)] = result
    return results



def main(unused_argv):
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # load training and eval data
    X,y = utils_parent.load_mnist(args.dataset)
    results_random_ressample = cross_validation(X,y,config.num_clusters,args)
    results_shifted = cross_validation_for_clustered_data(X,y,config.data_path,config.num_labels,config.num_clusters,args)
    print("***********random************")
    print(results_random_ressample)
    print("***********shifted************")
    print(results_shifted)
    utils_parent.write_results_convnet_to_csv("results_random.csv",results_random_ressample)
    utils_parent.write_results_convnet_to_csv("results_cluster.csv",results_shifted)




if __name__ == "__main__":
    tf.app.run()
