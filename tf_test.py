# ??Tensorflow????????

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
batch_size = 8

w1 = tf.Variable(tf.random_normal([1, 10], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([10, 30], stddev=1, seed=1))
w3 = tf.Variable(tf.random_normal([30, 1], stddev=1, seed=1))
b1 = tf.Variable(tf.random_normal([10], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([30], stddev=1, seed=1))
b3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))


x = tf.placeholder(tf.float32, shape=(None, 1), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)+b1
b = tf.nn.relu(a)
c = tf.matmul(b, w2) + b2
d = tf.nn.relu(c)
y = tf.matmul(d, w3)+b3

mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(0.001).minimize(mse)

dataset_size = 128
X: np.array = np.linspace(-5, 5, dataset_size)
Y: np.array = X**2+np.random.rand(dataset_size)*5

X = X.reshape(dataset_size, 1)
Y = Y.reshape(dataset_size, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    STEPS = 30000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_mse = sess.run(
                mse, feed_dict={x: X, y_: Y})
            print("Step:", i, "Loss:", total_mse)

    output = (sess.run(y, feed_dict={x: X, y_: Y}))

    plt.scatter(X, Y, color='y')
    plt.plot(X, output)
    plt.show()
