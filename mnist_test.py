# 使用Tensorflow的CNN网络识别MNIST手写数字


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Hyper Parameters
BATCH_SIZE = 128
LR_BASE = 0.0002
LR_DECAY = 0.99
TRAIN_STEP = 6000

mnist = input_data.read_data_sets(
    "E:/Test/Tensorflow/MNIST_data", one_hot=True)

# plt.imshow(mnist.train.images[123].reshape(28, 28), cmap='gray')
# plt.show()
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=1, seed=1))
w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=1, seed=1))
b1 = tf.Variable(tf.random_normal([32], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([64], stddev=1, seed=1))

w_fc1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024], stddev=1, seed=1))
w_fc2 = tf.Variable(tf.random_normal([1024, 10], stddev=1, seed=1))
b_fc1 = tf.Variable(tf.random_normal([1024], stddev=1, seed=1))
b_fc2 = tf.Variable(tf.random_normal([10], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 10), name='y-input')

conv1 = tf.nn.relu(tf.nn.conv2d(
    x, w1, strides=[1, 1, 1, 1], padding='SAME')+b1)
pool1 = tf.nn.max_pool2d(conv1, ksize=[1, 2, 2, 1], strides=[
                         1, 2, 2, 1], padding='SAME')

conv2 = tf.nn.relu(tf.nn.conv2d(
    pool1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
pool2 = tf.nn.max_pool2d(conv2, ksize=[1, 2, 2, 1], strides=[
                         1, 2, 2, 1], padding='SAME')

pool2 = tf.reshape(pool2, [-1, 7*7*64])
fc1 = tf.nn.relu(tf.matmul(pool2, w_fc1) + b_fc1)
y = tf.matmul(fc1, w_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=y, labels=tf.argmax(y_, 1)))
train = tf.train.AdamOptimizer(LR_BASE).minimize(cross_entropy)

dataset_size = mnist.train.num_examples
X = mnist.train.images.reshape([-1, 28, 28, 1])
Y = mnist.train.labels

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(TRAIN_STEP):
        start = (i*BATCH_SIZE) % dataset_size
        end = min(start+BATCH_SIZE, dataset_size)

        sess.run(train, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 100 == 0:
            total_loss = sess.run(
                cross_entropy, feed_dict={x: X[0:200], y_: Y[0:200]})
            print("Step:", i, "Loss:", total_loss)

    output = (sess.run(tf.argmax(y, 1), feed_dict={
        x: X[0:1000], y_: Y[0:1000]}))
    for i in range(5):
        print("Label:", output[600+i])
        plt.imshow(X[600+i].reshape(28, 28))
        plt.show()

    right_cnt = 0
    for i in range(1000):
        if output[i] == np.argmax(Y[i]):
            right_cnt += 1
    print("Accuracy:", right_cnt/1000)
