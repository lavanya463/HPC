#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_loss(y_data,filename):
  fig = plt.figure()
  plt.plot(y_data)
  #plt.yscale('log')
  plt.ylabel('Loss Function Value')
  plt.xlabel('Iteration Number')
  plt.show()
  fig.savefig("Ex1_curves/"+filename, dpi=fig.dpi)


# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.01)
#optimizer = tf.train.AdamOptimizer(0.01)
optimizer = tf.train.AdagradOptimizer(0.01)
#optimizer = tf.train.AdadeltaOptimizer(0.01)
#optimizer = tf.train.RMSPropOptimizer(0.01)
optimizer = tf.train.FtrlOptimizer(0.01)

file_name = "ag_0.01.png"

train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

loss_list = list()
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})
  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
  loss_list.append(float(curr_loss))

plot_loss(loss_list,file_name)








