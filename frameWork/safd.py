import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def shuffle(X, T, C):
	s = np.arange(len(X))
	np.random.shuffle(s)
	return np.array(X[s]), np.array(T[s]), np.array(C[s])


# parameters setting

time_steps = 22
n_input = 5
n_classes = 1

num_units = 128
learning_rate = .03

# n_samples = tf.placeholder(tf.int32)
batch_size = 128
# convert lstm units to class probability.
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))
para_list = [out_weights, out_bias]

# input placeholder
X = tf.placeholder(tf.float32, [None, time_steps, n_input])
T = tf.placeholder(tf.int32, shape=(None,))
C = tf.placeholder(tf.int32, shape=(None,))

# [batch_size, n_steps, n_input] to "time_steps" number of [batch_size, n_input] tensors
input = tf.unstack(X, time_steps, axis=1)

# define the network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

logits_series = [tf.matmul(state, out_weights) + out_bias for state in outputs] #Broadcasted addition
hazard_series = tf.convert_to_tensor([tf.nn.sigmoid(logits) for logits in logits_series])


# convert tensor to iterator by 'unstack'.
T_iter = tf.unstack(T, batch_size, axis=0)
C_iter = tf.unstack(C, batch_size, axis=0)
H_iter = tf.unstack(
                tf.reshape(
                    tf.transpose(hazard_series,(1, 0, 2)),[batch_size, time_steps]
                ),
                batch_size, axis=0)

#loss
loss_mle = tf.reduce_mean(
                [tf.reduce_sum(tf.cast(H_i[0:T_i-1],tf.float32))
                          -tf.multiply(tf.cast(C_i,tf.float32), tf.log(H_i[T_i-1]))
                                                for H_i, T_i, C_i in zip(H_iter, T_iter, C_iter)]
                )



#train step
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_mle, var_list=para_list)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_mle, var_list=para_list)

# Input
dest_path = "../Data/"

X_train = np.load(dest_path + "X_train.npy")
T_train = np.load(dest_path + "T_train.npy")
C_train = np.load(dest_path + "C_train.npy")

X_test = np.load(dest_path + "X_test.npy")
T_test = np.load(dest_path + "T_test.npy")
C_test = np.load(dest_path + "C_test.npy")


sess = tf.Session()
sess.run(tf.global_variables_initializer())


q = np.divide(len(X_train), batch_size)

for n_epoch in range(100):

    X_sfl, T_sfl, C_sfl = shuffle(X_train, T_train, C_train)

    for n_batch in range(q):

        _, _loss = sess.run([train_step, loss_mle],feed_dict={
                                X:X_sfl[n_batch*batch_size:(n_batch+1)*batch_size],
                                T:T_sfl[n_batch*batch_size:(n_batch+1)*batch_size],
                                C:C_sfl[n_batch*batch_size:(n_batch+1)*batch_size]})
        # print n_batch

    print "epoch %s: %s"%(n_epoch, _loss)
    n_epoch += 1

exit(0)











































































# _, _hazard = sess.run([train_step, losses], feed_dict={
#                         X: X_train,
#                         T: T_train,
#                         C: C_train})
#
# print _hazard


# sequential hazards
# output_flat = tf.reshape(outputs, [-1, num_units])
# hazard = tf.reshape(
#     tf.sigmoid(
#         tf.matmul(output_flat, out_weights) + out_bias
#     ),
#     [-1, time_steps, n_classes]
# )