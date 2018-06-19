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

num_units = 128
learning_rate = .00003

batch_size = 128
sigma = 3
theta = 0.8

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
# train_mle_rank = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=para_list)
train_mle_rank = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=para_list)


# Input
dest_path = "../Data/"


# train setting
X_train = np.load(dest_path + "X_train.npy")
T_train = np.load(dest_path + "T_train.npy")
C_train = np.load(dest_path + "C_train.npy")

acc_pair = json.load(open("../Data/acc_pair.json", "r"))
pair_list = np.array([[int(i), j] for i, v in acc_pair.items() for j in v if j < 3000])

X_train_event = list()
T_train_event = list()
for i, c in enumerate(C_train):
    if c == 1:
        X_train_event.append(X_train[i])
        T_train_event.append(T_train[i])

X_train_event, T_train_event = np.array(X_train_event), np.array(T_train_event)

def usr2obsT(T):
    u2T = dict()
    for i, t in enumerate(T):
        u2T[i] = t
    return u2T

u2T = usr2obsT(T_train_event)

i_bases = pair_list[:,0]
j_bases = pair_list[:,1]

i_off = [u2T[i]-1 for i in i_bases]

i_index = i_bases*X_train_event.shape[1] + i_off
j_index = j_bases*X_train_event.shape[1] + i_off



# test setting
X_test = np.load(dest_path + "X_test.npy")
T_test = np.load(dest_path + "T_test.npy")
C_test = np.load(dest_path + "C_test.npy")


X_test_event, T_test_event =  list(), list()
for i, c in enumerate(C_test):
    if c == 1:
        X_test_event.append(X_test[i])
        T_test_event.append(T_test[i])
X_test_event, T_test_event = np.array(X_test_event), np.array(T_test_event)

Acc_pair_test = list()
for i, t_i in enumerate(T_test_event):
    for j, t_j in enumerate(T_test_event):
        if j <= i:
            continue
        if t_i < t_j:
            Acc_pair_test.append([i,j])
        elif t_i > t_j:
            Acc_pair_test.append([j, i])

Acc_pair_test = np.array(Acc_pair_test)


print "-----------------------", Acc_pair_test.shape

# training & testing
sess = tf.Session()
sess.run(tf.global_variables_initializer())
q = np.divide(len(X_train), batch_size)


for n_epoch in range(10000):

    ran_ind = ran_seed(i_index.shape[0])
    i_index = i_index[ran_ind]
    j_index = j_index[ran_ind]

    for n_batch in range(q):

        _, _loss_rank = sess.run([train_rank, loss_rank],feed_dict={
                                X: X_train,
                                i_ind: i_index[n_batch * 10000:(n_batch + 1) * 10000],
                                j_ind: j_index[n_batch * 10000:(n_batch + 1) * 10000]
                                })

    # print "epoch %s: %s" % (n_epoch, _loss_rank)


    _H_test = np.array(
        sess.run([H], feed_dict={X:X_test_event})
    )
    _H_test = _H_test.reshape(_H_test.shape[1],_H_test.shape[2])


    # threshold
    # T_pred = list()
    # for hs in _H_test:
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
    for hs in _H_test:
        T_pred.append(np.argmax(hs)+1)

    # MAE: Mean absolute error
    # mse = np.mean(np.abs(T_test_event-T_pred))
    # print "epoch: %s"%n_epoch, mse, np.mean(T_pred), np.mean(T_test_event)

    # CI
    # ci_count = (T_pred[Acc_pair_test[:,1]]>T_pred[Acc_pair_test[:,0]]).astype(int)

    ci_count = 0
    for p in Acc_pair_test:
        if T_pred[p[1]] > T_pred[p[0]]:
            ci_count += 1
    # print ci_count
    CI = ci_count/float(Acc_pair_test.shape[0])

    print CI

    # exit(0)




exit(0)
