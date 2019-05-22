#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt




#read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0



#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#optimizer = tf.train.AdagradOptimizer(0.7).minimize(cross_entropy)
#optimizer = tf.train.AdadeltaOptimizer(9).minimize(cross_entropy)
#optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cross_entropy)
#optimizer = tf.train.FtrlOptimizer(0.5).minimize(cross_entropy)

file_name = "A_0.01.png"
#optimizer = tf.train.AdamOptimizer(0.01)
#optimizer = tf.train.AdagradOptimizer(0.01)
#optimizer = tf.train.AdadeltaOptimizer(0.01)
#optimizer = tf.train.RMSPropOptimizer(0.01)
#optimizer = tf.train.FtrlOptimizer(0.01)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

avg_set = list()
epoch_set = list()


def plot_loss(y_data,filename):
  fig = plt.figure()
  plt.plot(y_data)
  #plt.yscale('log')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.show()
  # fig.savefig("Ex2_curves/"+filename, dpi=fig.dpi)


#TRAINING PHASE
print("TRAINING")
for i in range(500):
  avg_cost = 0
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
  avg_cost += sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
  print ("Epoch:{} cost={}".format(i+1, avg_cost))
  avg_set.append(avg_cost)
  epoch_set.append(i+1)

plot_loss(avg_set,file_name)



#CHECKING THE ERROR
print("ERROR CHECK")

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))


