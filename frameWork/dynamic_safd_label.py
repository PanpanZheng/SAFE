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
time_steps = 22
n_classes = 1

num_units = 32
learning_rate = .001

sigma = 2
theta = 0.5

sur_thrld = 0.5

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
dest_path = "../Data2/"

# X_train = np.load(dest_path + "X_train_diff.npy")
# T_train = np.load(dest_path + "T_train_diff.npy")
# C_train = np.load(dest_path + "C_train_diff.npy")
#
# X_test = np.load(dest_path + "X_test_diff.npy")
# T_test = np.load(dest_path + "T_test_diff.npy")
# C_test = np.load(dest_path + "C_test_diff.npy")


X_train = np.load(dest_path + "X_train_np.npy")
T_train = np.load(dest_path + "T_train.npy")
C_train = np.load(dest_path + "C_train.npy")

X_test = np.load(dest_path + "X_test_np.npy")
T_test = np.load(dest_path + "T_test.npy")
C_test = np.load(dest_path + "C_test.npy")


session = tf.Session()
session.run(tf.global_variables_initializer())


num_epoches = 1000

for n_epoch in range(num_epoches):

    for n in range(time_steps):

        if n == 0:
            continue
        bat_X = X_train[T_train == (n+1)].tolist()
        bat_C = C_train[T_train == (n+1)].tolist()
        bat_size = np.sum(T_train == (n+1)).astype(int)

        _, _mle_loss, _lambdas = session.run([train_mle_op, mle_loss, lambdas],feed_dict={
                                X: bat_X,
                                C: bat_C,
                                mle_index: (np.vstack((np.arange(bat_size), np.zeros(bat_size)+n)).T).astype(int).tolist(),
                                batch_size: bat_size
                                # X: X_train[T_train == (n+1)].tolist(),
                                # C: C_train[T_train == (n+1)].tolist(),
                                # mle_index: (np.vstack((np.arange(np.sum(T_train == (n+1))), np.zeros(np.sum(T_train == (n+1)))+n)).T).astype(int).tolist(),
                                # batch_size: np.sum(T_train == (n+1)).astype(int)


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


_lambdas_test = list()
for n in range(time_steps):
    print "=== %s"%(n+1)
    if n == 0:
        continue
    tri_mat = upp_tri_mat(np.zeros((n + 1, n + 1)))
    H, S = session.run([lambdas, survivals],
                       feed_dict={X: X_test[T_test == (n + 1)].tolist(),
                                  batch_size: np.sum(T_test == (n + 1)).astype(int),
                                  mask: tri_mat})

    H, S = np.array(H), np.array(S)
    for h, s in zip(H,S):
        print h, ": ", s
    # print H.shape, S.shape
    # exit(0)
    # H = H.reshape(H.shape[1], H.shape[2])
    # _lambdas_test.append(H)

# survivals_test = list()
# X_group = list()
# for i, _lambdas_group in enumerate(_lambdas_test):
#     X_group.extend(X_test[T_test==(i+1)].tolist())
#     for H in _lambdas_group:
#         survivals_test.append(lambda2Survival(H))

# for s, x in zip(survivals_test, X_group):
#     print s, "  ", x
exit(0)

































# censor_detect_test = list()
# suspended_candidate_survival_test = list()
# suspended_candidate_survival_time_test = list()
# for j, survival in enumerate(survivals_test):
#     flag = True
#     for i in range(time_steps-1):
#         if survival[i] <= sur_thrld:
#             censor_detect_test.append(i+1)
#             suspended_candidate_survival_test.append(survival)
#             suspended_candidate_survival_time_test.append(Sur_Time_test[j])
#             flag = False
#             break
#     if flag:
#         censor_detect_test.append(time_steps)
#
# censor_detected_rate_test = np.sum((np.array(censor_detect_test)==23)==(C_test==0))/float(np.array(censor_detect_test).shape[0])
#
# sus_detect_test = list()
# for j, survival in enumerate(suspended_candidate_survival_test):
#     if suspended_candidate_survival_time_test[j] == 23:
#         continue
#     flag = True
#     for i in range(suspended_candidate_survival_time_test[j]):
#         if survival[i] <= sur_thrld:
#             sus_detect_test.append(i+1)
#             flag = False
#             break
#     if flag:
#         sus_detect_test.append(-1)
#
# acc_valid_test = np.sum(np.array(sus_detect_test)!=-1)/float(np.array(sus_detect_test).shape[0])
#
# print "censor_detect_rate: ", censor_detected_rate_test
# print "suspend_detect_rate: ", acc_valid_test