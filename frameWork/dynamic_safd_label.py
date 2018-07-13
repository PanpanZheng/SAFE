import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
import os
sys.path.append("../")

from safdKit import upp_tri_mat, ran_seed, concordance_index, acc_pair_wtte, in_interval,pick_up_pair, acc_pair, cut_seq, cut_seq_last, cut_seq_0, cut_seq_mean, lambda2Survival
from sklearn.metrics import classification_report, accuracy_score

# parameters setting
n_input = 5
time_steps = 21
n_classes = 1

num_units = 32
learning_rate = .001

sigma = 2
theta = 0.5

sur_thrld = 0.3

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)

X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
C = tf.placeholder(tf.float32, shape=[None,], name='C')
mle_index = tf.placeholder(tf.int32, shape=[None,2], name='mle_time')
batch_size = tf.placeholder(tf.int32, name='batch_size')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

lambdas = tf.reshape(
            tf.nn.softplus(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
            # tf.nn.sigmoid(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
            [batch_size, -1])

mle_loss = tf.reduce_mean(
    tf.subtract(
                tf.reduce_sum(lambdas, axis=1),
                tf.multiply(tf.log(tf.subtract(tf.exp(tf.gather_nd(lambdas, mle_index)), 1)) ,C),
                # tf.multiply(tf.log(tf.subtract(tf.exp(tf.reduce_sum(lambdas, axis=1)), 1)) ,C)
    )
)

y_true = tf.placeholder(tf.int32, shape=[None, None], name="y_real")
mask = tf.placeholder(tf.float32, shape=[None,None], name='upper_matrix')
survivals = tf.exp(-tf.matmul(lambdas, mask))
supervised_loss = tf.losses.mean_squared_error(y_true, survivals)


#for mle
train_mle = tf.train.AdamOptimizer(learning_rate)
m_gvs = train_mle.compute_gradients(mle_loss)
m_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in m_gvs]
train_mle_op = train_mle.apply_gradients(m_capped_gvs)


#for supervised

train_supervised = tf.train.AdamOptimizer(learning_rate)
s_gvs = train_supervised.compute_gradients(supervised_loss)
s_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in s_gvs]
train_supervised_op = train_supervised.apply_gradients(s_capped_gvs)



# load data
dest_path = "../Data3/"
# dest_path = "../Data2/"

X_train = np.load(dest_path + "X_train_var.npy")
T_train = np.load(dest_path + "T_train_var.npy")
C_train = np.load(dest_path + "C_train_var.npy")

X_test = np.load(dest_path + "X_test_var.npy")
T_test = np.load(dest_path + "T_test_var.npy")
C_test = np.load(dest_path + "C_test_var.npy")

# X_train = np.load(dest_path + "X_train_diff.npy")
# T_train = np.load(dest_path + "T_train_diff.npy")
# C_train = np.load(dest_path + "C_train_diff.npy")
#
# X_test = np.load(dest_path + "X_test_diff.npy")
# T_test = np.load(dest_path + "T_test_diff.npy")
# C_test = np.load(dest_path + "C_test_diff.npy")

# X_train = np.load(dest_path + "X_train_np.npy")
# T_train = np.load(dest_path + "T_train.npy")
# C_train = np.load(dest_path + "C_train.npy")
#
# X_test = np.load(dest_path + "X_test_np.npy")
# T_test = np.load(dest_path + "T_test.npy")
# C_test = np.load(dest_path + "C_test.npy")

session = tf.Session()
session.run(tf.global_variables_initializer())


num_epoches = 300

for n_epoch in range(num_epoches):

    for n in range(time_steps):

        if n == 0:
            continue
        bat_X = X_train[T_train == (n+1)].tolist()
        bat_C = C_train[T_train == (n+1)].tolist()
        bat_size = np.sum(T_train == (n+1)).astype(int)
        if bat_size == 0:
            continue

        _, _mle_loss, _lambdas = session.run([train_mle_op, mle_loss, lambdas],feed_dict={
                                X: bat_X,
                                C: bat_C,
                                mle_index: (np.vstack((np.arange(bat_size), np.zeros(bat_size)+n)).T).astype(int).tolist(),
                                batch_size: bat_size
        })

        tri_mat = upp_tri_mat(np.zeros((n+1, n+1)))
        if bat_C[0] == 1:
            labels = np.zeros((bat_size,n+1))
        else:
            labels = np.ones((bat_size,n+1))

        _, _supervised_loss, _survivals = session.run([train_supervised_op, supervised_loss, survivals], feed_dict={
                                X: bat_X,
                                C: bat_C,
                                mle_index: (np.vstack((np.arange(bat_size), np.zeros(bat_size) + n)).T).astype(int).tolist(),
                                batch_size: bat_size,
                                mask:tri_mat,
                                y_true:labels
        })


    print "epoch: ", n_epoch, _mle_loss, _supervised_loss
    # print "epoch: ", n_epoch, _mle_loss
    # print "epoch: ", n_epoch, _supervised_loss

Survival = list()
for n in range(time_steps):
    if n == 0:
        continue
    tri_mat = upp_tri_mat(np.zeros((n+1, n+1)))
    bat_size = np.sum(T_test == (n+1)).astype(int)
    if bat_size == 0:
        continue
    _survivals = session.run([survivals],
                               feed_dict={X: X_test[T_test == (n+1)].tolist(),
                                          batch_size: bat_size,
                                          mask: tri_mat})

    _survivals = np.array(_survivals)
    _survivals = _survivals.reshape(_survivals.shape[1],_survivals.shape[2])
    Survival.extend(_survivals)

unc_cen = list()
for ss in Survival:
    event_flag = False
    for s in ss:
        if s <= sur_thrld:
            event_flag = True
            break
    if event_flag:
        unc_cen.append(1)
    else:
        unc_cen.append(0)

unc_cen = np.array(unc_cen)
unc_cen_acc = accuracy_score(C_test, unc_cen)
unc_det_acc = np.sum(unc_cen[C_test == 1])/float(np.sum(np.array(unc_cen)==1))
print "censor or uncensor ? : ", unc_cen_acc
print "given uncensor above, the true uncensor accuracy: ", unc_det_acc


exit(0)