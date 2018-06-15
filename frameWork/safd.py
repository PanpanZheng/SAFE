import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json

import sys
sys.path.append("../")
from safdKit import ran_seed, concordance_index


# parameters setting
n_input = 5
time_steps = 22
n_classes = 1

num_units = 16
learning_rate = .03

batch_size = 128
sigma = 0.1
theta = 0.5

# convert lstm units to class probability.
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))
para_list = [out_weights, out_bias]

# input placeholder
X = tf.placeholder(tf.float32, [None, time_steps, n_input])
X_bat = tf.placeholder(tf.float32, [None, time_steps, n_input])
C = tf.placeholder(tf.float32, shape=(None,))

# [batch_size, n_steps, n_input] to "time_steps" number of [batch_size, n_input] tensors
input = tf.unstack(X,time_steps,axis=1)
input_bat = tf.unstack(X_bat,time_steps,axis=1)

# define the network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")
outputs_bat, _ = rnn.static_rnn(lstm_layer, input_bat, dtype="float32")

H = tf.reshape(
                tf.nn.sigmoid(
                             tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)
                                                                                    + out_bias),
                                                                                            [-1,time_steps])
H_bat = tf.reshape(
                tf.nn.sigmoid(
                             tf.matmul(tf.reshape(outputs_bat,[-1,num_units]),out_weights)
                                                                                    + out_bias),
                                                                                            [-1,time_steps])

#loss
mle_mask = tf.placeholder(tf.float32, [None, time_steps])
mle_index = tf.placeholder(tf.int32, shape=(None,))
loss_mle = tf.subtract(
                tf.reduce_sum(
                    tf.multiply(H_bat, mle_mask)),   # H_bat: partial,  mle_mask: partial.
                                    tf.reduce_sum(tf.multiply(
                                                tf.log(
                                                    tf.gather(tf.reshape(H, [-1]), mle_index)   #  hazard_series: whole,  mle_index partial
                                                ),C  # C partial.
                                            )))

# loss_rank
i_ind = tf.placeholder(tf.int32, shape=(None,))
j_ind = tf.placeholder(tf.int32, shape=(None,))
loss_rank = tf.reduce_sum(
                tf.exp(
                    tf.divide(
                        tf.subtract(
                            tf.gather(tf.reshape(H, [-1]), j_ind),  # hazard_series, whole, j_ind partial.
                            tf.gather(tf.reshape(H, [-1]), i_ind)   # hazard_series, whole, i_ind partial.
                        ), sigma)
                )
            )

#train step
train_mle = tf.train.AdamOptimizer(learning_rate).minimize(loss_mle, var_list=para_list)
train_rank = tf.train.AdamOptimizer(learning_rate).minimize(loss_rank, var_list=para_list)
loss = theta*loss_mle + (1-theta)*loss_rank
train_mle_rank = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=para_list)
# train_mle_rank = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=para_list)


# Input
dest_path = "../Data/"

X_train = np.load(dest_path + "X_train.npy")
T_train = np.load(dest_path + "T_train.npy")
C_train = np.load(dest_path + "C_train.npy")

X_mask = np.zeros([X_train.shape[0], X_train.shape[1]])
for v, t in zip(X_mask, T_train):
    for i in range(t):
        v[i] = 1
X_bases = np.multiply(np.arange(X_train.shape[0]), X_train.shape[1])
X_index = np.add(X_bases, np.subtract(T_train,1))



acc_pair = json.load(open("../Data/acc_pair.json", "r"))
pair_list = np.array([[int(i), j] for i, v in acc_pair.items() for j in v])

def usr2obsT(T):
    u2T = dict()
    for i, t in enumerate(T):
        u2T[i] = t
    return u2T

u2T = usr2obsT(T_train)

i_bases = pair_list[:,0]
j_bases = pair_list[:,1]

i_off = [u2T[i]-1 for i in i_bases]
# j_off = [u2T[j]-1 for j in j_bases]

i_index = i_bases*X_train.shape[1] + i_off
j_index = j_bases*X_train.shape[1] + i_off

X_test = np.load(dest_path + "X_test.npy")
T_test = np.load(dest_path + "T_test.npy")
C_test = np.load(dest_path + "C_test.npy")

X_event, T_event, X_censor, T_censor = list(), list(), list(), list()
for i, c in enumerate(C_test):
    if c == 1:
        X_event.append(X_test[i])
        T_event.append(T_test[i])
    else:
        X_censor.append(X_test[i])
        T_censor.append(T_test[i])
X_event, T_event, X_censor, T_censor = np.array(X_event), np.array(T_event), np.array(X_censor), np.array(T_censor)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
q = np.divide(len(X_train), batch_size)


for n_epoch in range(100):

    ran_ind = ran_seed(i_index.shape[0])
    i_index = i_index[ran_ind]
    j_index = j_index[ran_ind]

    for n_batch in range(q):

        # _, _loss_mle = sess.run([train_mle, loss_mle],feed_dict={
        #                                 X: X_train,
        #                                 X_bat: X_train[n_batch * batch_size:(n_batch + 1) * batch_size],
        #                                 C: C_train[n_batch * batch_size:(n_batch + 1) * batch_size],
        #                                 mle_mask: X_mask[n_batch * batch_size:(n_batch + 1) * batch_size],
        #                                 mle_index: X_index[n_batch * batch_size:(n_batch + 1) * batch_size]})
        #
        # _, _loss_rank = sess.run([train_rank, loss_rank],feed_dict={
        #                         X: X_train,
        #                         i_ind: i_index[n_batch * 10000:(n_batch + 1) * 10000],
        #                         j_ind: j_index[n_batch * 10000:(n_batch + 1) * 10000]
        #                         })
        #
        _, _loss, _loss_mle, _loss_rank = sess.run([train_mle_rank, loss, loss_mle, loss_rank],feed_dict={
                                            X: X_train,
                                            X_bat: X_train[n_batch*batch_size:(n_batch+1)*batch_size],
                                            C: C_train[n_batch*batch_size:(n_batch+1)*batch_size],
                                            mle_mask: X_mask[n_batch*batch_size:(n_batch+1)*batch_size],
                                            mle_index: X_index[n_batch*batch_size:(n_batch+1)*batch_size],
                                            i_ind: i_index[n_batch*10000:(n_batch+1)*10000],
                                            j_ind: j_index[n_batch*10000:(n_batch+1)*10000]})

    # print "epoch %s: %s" % (n_epoch, _loss_mle)
    #
    # print "epoch %s: %s" % (n_epoch, _loss_rank)

    # print "epoch %s: %s %s" % (n_epoch, _loss_mle, _loss_rank)

    # print "epoch %s: %s %s %s" % (n_epoch, _loss, _loss_mle, _loss_rank)


    _H = np.array(
        sess.run([H], feed_dict={X:X_event})
    )
    _H = _H.reshape(_H.shape[1],_H.shape[2])


    # threshold
    # T_pred = list()
    # for hs in _H:
    #     flag = True
    #     for i, h in enumerate(hs):
    #         if h > .35:
    #             T_pred.append(i+1)
    #             flag = False
    #             break
    #     if flag:
    #         T_pred.append(22)


    # max hazard
    T_pred = list()
    for hs in _H:
        T_pred.append(np.argmax(hs)+1)


    print
    print T_event[0:100]
    print "-------------------"
    print T_pred[0:100]
    print
    mse = np.mean(np.abs(T_event-T_pred))
    print "epoch: %s"%n_epoch, mse, np.mean(T_pred), np.mean(T_event)


exit(0)
