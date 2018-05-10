# Adapted from Deep MNIST for Experts

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def convolution2d(image, weight):
  return tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #  interleave operations which build a computation graph with operations(TensorFlow operations arranged into a graph of nodes) that run the graph.
    sess = tf.InteractiveSession()

    input_image = tf.placeholder(tf.float32, shape=[None, 784])
    # Each row is a 10-dimensional vector indicating the digit class(0-9) the MNIST image belongs to
    target_output_class = tf.placeholder(tf.float32, shape=[None, 10])

    current_image = tf.reshape(input_image, [-1, 28, 28, 1])

    final_conv, keep_prob = cnn_model(current_image)
    # Loss function
    cross_entropy = -tf.reduce_sum(target_output_class * tf.log(final_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(final_conv, 1), tf.argmax(target_output_class, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                input_image: batch[0], target_output_class: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={input_image: batch[0], target_output_class: batch[1], keep_prob: 0.5})

    save_path = saver.save(sess, "C:/Users/Amanda/PycharmProjects/jaarProjek/cnn_trainingModel.ckpt")
    print("Model saved in file: ", save_path)

    print("test accuracy %g" % accuracy.eval(feed_dict={
        input_image: mnist.test.images, target_output_class: mnist.test.labels, keep_prob: 1.0}))


def cnn_model(img):

    current_image = tf.reshape(img, [-1, 28, 28, 1])

    # 32 features for each 5x5 patch
    weight_conv1 = weight_variable([5, 5, 1, 32])
    bias_conv1 = bias_variable([32])

    conv1 = tf.nn.relu(convolution2d(current_image, weight_conv1) + bias_conv1)
    pool1 = max_pool(conv1)

    weight_conv2 = weight_variable([5, 5, 32, 64])
    bias_conv2 = bias_variable([64])

    conv2 = tf.nn.relu(convolution2d(pool1, weight_conv2) + bias_conv2)
    pool2 = max_pool(conv2)

    weight_fully_connected_layer1 = weight_variable([7 * 7 * 64, 1024])
    bias_fully_connected_layer1 = bias_variable([1024])

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fully_connected_layer1 = tf.nn.relu(
        tf.matmul(pool2_flat, weight_fully_connected_layer1) + bias_fully_connected_layer1)

    keep_prob = tf.placeholder(tf.float32)
    fully_connected_layer1_dropout = tf.nn.dropout(fully_connected_layer1, keep_prob)

    weight_fully_connected_layer2 = weight_variable([1024, 10])
    bias_fully_connected_layer2 = bias_variable([10])

    final_conv = tf.nn.softmax(
        tf.matmul(fully_connected_layer1_dropout, weight_fully_connected_layer2) + bias_fully_connected_layer2)
    return final_conv, keep_prob


def buildTheCNN():
    build_cnn()

buildTheCNN()

