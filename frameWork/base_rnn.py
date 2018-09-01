import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import sys
sys.path.append("../")
from safdKit import prec_reca_F1, get_first_beat, early_det
from collections import defaultdict

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

np.random.seed(7)
before_steps = 5
n_classes = 1

num_units = 32
batch_size = 64
learning_rate = .001
time_steps = 21

# load data
if len(sys.argv) != 2:
    print "please add 1 parameter."
    exit(0)

if sys.argv[1] == "twitter":

    source_path = "../twitter/"
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


def minibatch(X, T, C, batch_size=16, n_input=8):

    minibatch_list_x = []
    minibatch_list_y = []
    minibatch_list_t = []
    minibatch_list_c = []
    # count = 0
    for t in np.unique(T):

        sub_X = X[np.where(T==t)]
        sub_C = C[np.where(T==t)]
        sub_T = T[np.where(T==t)]
        n_sub_batch = int(sub_X.shape[0]/batch_size)
        # count += n_sub_batch*batch_size
        for n in range(n_sub_batch):
            minibatch_list_x.append(np.asarray(list(sub_X[n*batch_size: (n+1)*batch_size])).reshape(batch_size,int(t),n_input))
            # minibatch_list_t.append(np.asarray(list(sub_T[n*batch_size: (n+1)*batch_size])))
            minibatch_list_t.append(t)
            minibatch_list_c.append(np.asarray(list(sub_C[n*batch_size: (n+1)*batch_size])).reshape(batch_size, 1))
            # minibatch_list_y.append(
            #     np.logical_not(
            #         np.asarray(list(sub_C[n*batch_size:(n+1)*batch_size])).reshape(batch_size, 1)
            #     ).astype(float)
            # )
    # assert(len(minibatch_list_x)==len(minibatch_list_y))
    minibatch_list_y = minibatch_list_c
    c = list(zip(minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c))
    random.shuffle(c)
    minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c = zip(*c)
    return minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c

mini_batch_size = 16
mini_batch_size_test_valid = 5
batch_x_train, batch_y_train, batch_t_train, batch_c_train = minibatch(X_train, T_train, C_train, mini_batch_size, n_input)
batch_x_test, batch_y_test, batch_t_test, batch_c_test = minibatch(X_test, T_test, C_test, mini_batch_size_test_valid, n_input)

# batch_x_train, batch_y_train, batch_t_train = minibatch(X_train, T_train, )
# batch_x_test, batch_y_test, batch_t_test = minibatch(X_test, T_test)
test_num = len(batch_x_test)*mini_batch_size_test_valid

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)

X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
y = tf.placeholder(tf.float32, shape=[None,1], name="y_real")
t = tf.placeholder(tf.int32, name='time_steps')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")
# Select last output.
last = tf.squeeze(tf.transpose (outputs, [0, 2, 1])[:,:,-1])
last = outputs[:,-1,:]

pred_logits = tf.matmul(last, out_weights) + out_bias

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logits, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

pred_y = tf.nn.sigmoid(pred_logits)
seq_pred_y = tf.reshape(
            tf.nn.sigmoid(tf.matmul(tf.reshape(outputs,[-1, num_units]), out_weights)+ out_bias),
            [-1, t])

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    for n_epoch in range(20):
        loss = []
        for batch_x, batch_y in zip(batch_x_train, batch_y_train):
            _, _cost = sess.run([optimizer, cost],feed_dict={X: batch_x, y: batch_y})
            loss.append(_cost)
        print("epoch: ", n_epoch, " loss: ", np.mean(loss))

    correct = 0
    early_correct = []
    yy = []
    pp = []
    early_detect_steps = defaultdict(list)
    early_detect_rate = defaultdict(list)
    early_detect_num = defaultdict(list)
    for batch_x, batch_y, batch_t in zip(batch_x_test, batch_y_test, batch_t_test):
        _pred_y_test, _seq_pred_y = sess.run([pred_y, seq_pred_y],feed_dict={X:batch_x, t:batch_t})
        pred_y_test = np.zeros(_pred_y_test.shape[0])
        pred_y_test[np.where(_pred_y_test.flatten()>0.5)] = 1
        correct += np.sum(pred_y_test==batch_y.flatten())

        # if int(batch_t) == 21:
        #     continue

        seq_events = _seq_pred_y[batch_y.flatten() == 1]

        if seq_events.shape[0] != 0:
            seq_pred_me_test = np.zeros((seq_events.shape[0], int(batch_t)))
            seq_pred_me_test[np.where(seq_events > 0.5)] = 1
            batch_me = np.asarray([np.ones(seq_events.shape[0]), ] * int(batch_t)).transpose()
            # fb = get_first_beat(seq_pred_me_test, batch_me)
            fb = early_det(seq_pred_me_test, batch_me)
            early_detect_steps[int(batch_t)].extend(np.multiply(np.ones(fb.shape[0]), int(batch_t)) - fb)


        # seq_pred_msa_test = np.zeros((_pred_y_test.shape[0],int(batch_t)))
        # seq_pred_msa_test[np.where(_seq_pred_y > 0.5)] = 1
        # batch_msa = np.asarray([batch_y.flatten(),]*int(batch_t)).transpose()
        #
        # fb = get_first_beat(seq_pred_msa_test, batch_msa)
        # early_detect_steps[int(batch_t)].append(
        #                                         np.mean(np.multiply(
        #                                             np.ones(fb.shape[0]),int(batch_t))-fb)
        #                                       )
        # early_detect_rate[int(batch_t)].append(np.divide(fb.shape[0],batch_msa.shape[0],dtype="float"))
        # early_detect_num[int(batch_t)].append(fb.shape[0])

        # _seq_pred_y = _seq_pred_y[:,:before_steps]
        seq_pred_y_test = np.zeros((_pred_y_test.shape[0], before_steps))
        seq_pred_y_test[np.where(_seq_pred_y[:,:before_steps]>0.5)] = 1
        batch_y = np.asarray([batch_y.flatten(),]*before_steps).transpose()
        early_correct.append(np.sum(seq_pred_y_test==batch_y, axis=0))
        yy.extend(batch_y)
        pp.extend(seq_pred_y_test)

    yy = np.asarray(yy)
    pp = np.asarray(pp)
    gt = yy
    pr = pp
    _precision, _recall, _F1 = prec_reca_F1(gt, pr)
    seq_corr_rate = np.sum(np.asarray(early_correct), axis=0)/float(test_num)

    print
    print
    print
    print "precision: ", _precision
    print "recall: ", _recall
    print "F1: ", _F1
    print "accuracy: ", seq_corr_rate

    # print "------------------------------------------------------"
    # for k, v in early_detect_steps.items():
    #     print k, ": ", v, np.mean(v), len(v)

exit(0)