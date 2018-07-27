import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import sys
sys.path.append("../")
from safdKit import minibatch, prec_reca_F1, get_first_beat
from collections import defaultdict

# load data
if len(sys.argv) != 2:
    print "please add 1 parameter."
    exit(0)

if sys.argv[1] == "twitter":

    source_path = "../diff_data_2/"
    X_train = np.load(source_path + "X_train_var.npy")
    T_train = np.load(source_path + "T_train_var.npy")
    C_train = np.load(source_path + "C_train_var.npy")

    X_test = np.load(source_path + "X_test_var.npy")
    T_test = np.load(source_path + "T_test_var.npy")
    C_test = np.load(source_path + "C_test_var.npy")

    X_valid = np.load(source_path + "X_valid_var.npy")
    T_valid = np.load(source_path + "T_valid_var.npy")
    C_valid = np.load(source_path + "C_valid_var.npy")

    n_input = 5

elif sys.argv[1] == "wiki":

    source_path = "../wiki/v8/"
    X_train = np.load(source_path + "X_train.npy")
    T_train = np.load(source_path + "T_train.npy")
    C_train = np.load(source_path + "C_train.npy")

    X_test = np.load(source_path + "X_test.npy")
    T_test = np.load(source_path + "T_test.npy")
    C_test = np.load(source_path + "C_test.npy")

    X_valid = np.load(source_path + "X_valid.npy")
    T_valid = np.load(source_path + "T_valid.npy")
    C_valid = np.load(source_path + "C_valid.npy")

    n_input = 8

else:
    print "parameter is not right, twitter or wiki."
    exit(0)

# print X_train.shape, T_train.shape, C_train.shape
# print X_test.shape, T_test.shape, C_test.shape
# print X_valid.shape, T_valid.shape, C_valid.shape
#
# exit(0)


# parameters setting
n_classes = 1
before_steps = 5

num_units = 32
learning_rate = .001

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)

X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
y_true = tf.placeholder(tf.int32, shape=[None,], name="y_real")
batch_size = tf.placeholder(tf.int32, name='batch_size')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

y_seq_pred = tf.reshape(
            tf.sigmoid(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
            [batch_size, -1])
y_pred = y_seq_pred[:,-1]

mse_loss = tf.losses.mean_squared_error(y_true, y_pred)

#for supervised
train_supervised = tf.train.AdamOptimizer(learning_rate)
s_gvs = train_supervised.compute_gradients(mse_loss)
s_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in s_gvs]
train_supervised_op = train_supervised.apply_gradients(s_capped_gvs)


session = tf.Session()
session.run(tf.global_variables_initializer())

mini_batch_size = 16
mini_batch_size_test_valid = 5
batch_x_train, batch_y_train, batch_t_train, batch_c_train = minibatch(X_train, T_train, C_train, mini_batch_size, n_input)
batch_x_test, batch_y_test, batch_t_test, batch_c_test = minibatch(X_test, T_test, C_test, mini_batch_size_test_valid, n_input)


test_num = len(batch_x_test)*mini_batch_size_test_valid

init = tf.global_variables_initializer()
with tf.Session() as sess:
    loss = []
    sess.run(init)
    for n_epoch in range(20):
        for batch_x, batch_y in zip(batch_x_train, batch_y_train):
            _, _cost = sess.run([train_supervised_op, mse_loss],feed_dict={X: batch_x, y_true: batch_y, batch_size:batch_x.shape[0]})
            loss.append(_cost)
        print("epoch: ", n_epoch, " loss: ", np.mean(loss))

    correct = 0
    early_correct = []

    yy = []
    pp = []

    early_detect_steps = defaultdict(list)
    # early_detect_rate = defaultdict(list)
    # early_detect_num = defaultdict(list)
    for batch_x, batch_y, batch_t, batch_c in zip(batch_x_test, batch_y_test, batch_t_test, batch_c_test):
        _pred_y_test, _seq_pred_y = sess.run([y_pred, y_seq_pred],feed_dict={X:batch_x, batch_size:batch_x.shape[0]})
        pred_y_test = np.zeros(_pred_y_test.shape[0])
        pred_y_test[np.where(_pred_y_test>0.5)] = 1
        correct += np.sum(pred_y_test == batch_y)

        # if int(batch_t[0]) == 21:
        #     continue

        seq_pred_msa_test = np.zeros((_pred_y_test.shape[0], int(batch_t[0])))
        seq_pred_msa_test[np.where(_seq_pred_y > 0.5)] = 1
        batch_msa = np.asarray([batch_y.flatten(),]*int(batch_t[0])).transpose()

        fb = get_first_beat(seq_pred_msa_test, batch_msa)
        early_detect_steps[int(batch_t[0])].append(
                                                np.mean(np.multiply(
                                                    np.ones(fb.shape[0]),int(batch_t[0]))-fb)
                                              )
        # early_detect_rate[int(batch_t[0])].append(np.divide(fb.shape[0],batch_msa.shape[0],dtype="float"))
        # early_detect_num[int(batch_t[0])].append(fb.shape[0])

        # _seq_pred_y = _seq_pred_y[:,:before_steps]
        seq_pred_y_test = np.zeros((_pred_y_test.shape[0], before_steps))
        seq_pred_y_test[np.where(_seq_pred_y[:,:before_steps] > 0.5)] = 1
        batch_y = np.asarray([batch_y,]*before_steps).transpose()
        early_correct.append(np.sum(seq_pred_y_test==batch_y, axis=0))
        yy.extend(batch_y)
        pp.extend(seq_pred_y_test)


    yy = np.asarray(yy)
    pp = np.asarray(pp)
    gt = np.logical_not(yy).astype(float)
    pr = np.logical_not(pp).astype(float)
    _precision, _recall, _F1 = prec_reca_F1(gt, pr)
    seq_corr_rate = np.sum(np.asarray(early_correct), axis=0)/float(test_num)

    print
    print
    print
    print "precision: ", _precision
    print "recall: ", _recall
    print "F1: ", _F1
    print "accuracy: ", seq_corr_rate

    print "------------------------------------------------------"
    coll_me = list()
    for k, v in early_detect_steps.items():
        coll_me.extend(v)

    print "mae: ", np.mean(coll_me)

exit(0)