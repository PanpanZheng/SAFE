import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np



# X = np.load("../Data/" + "X.npy")
# T = np.load("../Data/" + "T.npy")
# C = np.load("../Data/" + "C.npy")


mnist = input_data.read_data_sets("./mnist/", one_hot=True)

# import mnist dataset

time_steps = 28
num_units = 128
n_input = 28

learning_rate = .03

n_classes = 10
batch_size = 128


# weights and biases of appropriate shape to accomplish above task.

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# input image placeholder
x = tf.placeholder("float", [None, time_steps, n_input])

# input label placeholder
y = tf.placeholder("float", [None, n_classes])

# Now that we are receiving inputs of shape [batch_size, time_steps,n_input], we need to convert it into a list of tensor of shape
# [batch_size, n_inputs] of length time_steps so that it can be then fed to static_rnn.

# processing the input tensor from [batch_size, n_steps, n_input] to "time_steps" number of [batch_size, n_input] tensors
# input = tf.unstack(x,time_steps, 1)
input = tf.unstack(x,axis=1)

# Now we are ready to define our network. We will use one layer of BasicLSTMCell and make our static_rnn network out of it.

# define the network

lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# As we are considered only with input of last time step, we will generate our prediction out of it.

# Converting last output of dimension [batch_size, num_units] to [batch_size, n_classes] by out_weight multiplication

prediction = tf.matmul(outputs[-1], out_weights) + out_bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

# optimization
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# inistialize variables

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    iter = 1
    while iter < 800:
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, n_input))

        sess.run(opt, feed_dict={x:batch_x, y:batch_y})

        # _input = np.array(
        #             sess.run([input], feed_dict={x: batch_x, y: batch_y}))
        #
        #
        # _outputs = np.array(
        #             sess.run([outputs], feed_dict={x: batch_x, y: batch_y}))
        #
        #
        # _prediction = np.array(
        #             sess.run([prediction], feed_dict={x: batch_x, y: batch_y}))

        # _y= np.array(
        #             sess.run([y], feed_dict={x: batch_x, y: batch_y}))

        # print _input.shape
        # print _outputs.shape
        # print _prediction.shape
        # print _y.shape
        # exit(0)


        if iter%10 == 0:
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            los = sess.run(loss, feed_dict={x:batch_x,y:batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("_____________________________")
        iter = iter +1




    # calculating test accuracy

    test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels[:128]

    print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:test_data, y:test_label}))






