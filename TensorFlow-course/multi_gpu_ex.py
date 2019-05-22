import time
import read_inputs
import numpy as N

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib


#declare weights and biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def model(X, reuse=False):
    with tf.variable_scope('L1', reuse=reuse):

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(X, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #drop_out
        keep_prob = 0.5
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv


def plot_loss(x_data,y_data,filename):
  fig = plt.figure()
  plt.plot(x_data,y_data)
  plt.xticks(x_data)
  plt.yticks([int(y) for y in y_data])
  plt.ylabel('Training time (sec)')
  plt.xlabel('Batch Size')
  fig.savefig(filename, dpi=fig.dpi)


if __name__ == '__main__':

    learning_rate = 0.001
    times_with_gpu_data = list()

    gpu_num = 2
    batch_sizes = [4,8,12,16,20,24,28,32,36,40,44,48]

    for batch_size in batch_sizes:
        iter_num = int(50000/batch_size)

        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        Y = tf.placeholder(tf.float32, [None, 10])

        losses = []

        X_A = tf.split(X, int(gpu_num))
        Y_A = tf.split(Y, int(gpu_num))

        # this part of code is inspired from the repo https://github.com/golbin/TensorFlow-Multi-GPUs/blob/master/many-GPUs-MNIST.py
        for gpu_id in range(int(gpu_num)):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                    cost = tf.nn.softmax_cross_entropy_with_logits(
                                    logits=model(X_A[gpu_id], gpu_id > 0),
                                    labels=Y_A[gpu_id])

                    losses.append(cost)

        loss = tf.reduce_mean(tf.concat(losses, axis=0))


        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss, colocate_gradients_with_ops=True)  # Important!

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        #mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
        mnist = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
        data = mnist[0]

        #data layout changes since output should an array of 10 with probabilities
        real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
        for i in range ( N.shape(data[0][1])[0] ):
          real_output[i][data[0][1][i]] = 1.0


        #data layout changes since output should an array of 10 with probabilities
        real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
        for i in range ( N.shape(data[2][1])[0] ):
          real_check[i][data[2][1][i]] = 1.0

        start_time = time.time()

        for i in range(iter_num):

            batch_ini = batch_size*i
            batch_end = batch_size*i+batch_size

            batch_xs = data[0][0][batch_ini:batch_end]
            batch_ys = real_output[batch_ini:batch_end]
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)


            _, cost_val = sess.run([optimizer, loss],feed_dict={X: batch_xs, Y: batch_ys})

        end_time = time.time()
        time_taken = end_time - start_time
        print("--- Training time : {0} seconds /w {1} GPUs ---".format(time_taken, gpu_num))

        times_with_gpu_data.append(time_taken)

    print(times_with_gpu_data)

    #plot_loss(batch_sizes,times_with_gpu_data,"gpu"+str(gpu_num)+".png")


"""
# training time lists for different number of GPU's
times_with_gpu1_data = [59.296364307403564, 30.073482513427734, 22.40762233734131, 17.834802865982056, 15.236272811889648, 13.411693334579468, 11.891669750213623, 10.75471043586731,10.05471043586731, 9.53304147720337, 8.953336954116821, 8.406367063522339]
times_with_gpu2_data = [65.8672981262207, 32.739357233047485, 22.869844436645508, 16.86134099960327, 14.235670566558838, 12.415099143981934, 10.992003202438354, 9.869258165359497, 9.009258165359497, 8.35128664970398, 7.859222888946533, 7.2250449657440186]
times_with_gpu4_data = [71.68086051940918, 20.025123357772827, 13.301606178283691, 9.711214780807495, 8.226654052734375, 7.027905702590942]

"""
