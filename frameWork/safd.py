import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
import os
sys.path.append("../")

from safdKit import upp_tri_mat, ran_seed
from sklearn.metrics import classification_report, accuracy_score

# parameters setting
n_input = 5
# time_steps = 22
time_steps = 21
n_classes = 1

num_units = 32
learning_rate = .001

# sur_thrld = 0.3

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
                tf.multiply(tf.log(tf.subtract(tf.exp(tf.gather_nd(lambdas, mle_index)), 1)) ,C)
    )
)

y_true = tf.placeholder(tf.int32, shape=[None, None], name="y_real")
mask = tf.placeholder(tf.float32, shape=[None,None], name='upper_matrix')  # upper triangle matrix
survivals = tf.exp(-tf.matmul(lambdas, mask))
supervised_loss = tf.losses.mean_squared_error(y_true, survivals)
# supervised_loss =tf.losses.sigmoid_cross_entropy(y_true, survivals)


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


#for check points.
saver = tf.train.Saver()
save_dir = '../checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "best_validation")

# load data
source_path = "../diff_data/"

X_train = np.load(source_path + "X_train_var.npy")
T_train = np.load(source_path + "T_train_var.npy")
C_train = np.load(source_path + "C_train_var.npy")

X_test = np.load(source_path + "X_test_var.npy")
T_test = np.load(source_path + "T_test_var.npy")
C_test = np.load(source_path + "C_test_var.npy")

X_valid = np.load(source_path + "X_valid_var.npy")
T_valid = np.load(source_path + "T_valid_var.npy")
C_valid = np.load(source_path + "C_valid_var.npy")


s = ran_seed(X_train.shape[0])
X_train, T_train, C_train = X_train[s], T_train[s], C_train[s]
s = ran_seed(X_test.shape[0])
X_test, T_test, C_test = X_test[s], T_test[s], C_test[s]

# print X_train.shape, T_train.shape, C_train.shape
# print X_test.shape, T_test.shape, C_test.shape
# print X_valid.shape, T_valid.shape, C_valid.shape
# exit(0)

session = tf.Session()
session.run(tf.global_variables_initializer())

global best_validation_accuracy
global best_thrld
global last_improvement
global require_improvement

best_validation_accuracy = 0.0
best_thrld = 0.0
last_improvement = 0
require_improvement = 50
num_epoches = 500

for n_epoch in range(num_epoches):

    for n in range(time_steps):

        if n == 0:
            continue
        bat_X = X_train[T_train==(n+1)].tolist()
        bat_C = C_train[T_train==(n+1)].tolist()
        bat_size = np.sum(T_train==(n+1)).astype(int)
        if bat_size == 0:
            continue
        _, _mle_loss = session.run([train_mle_op, mle_loss],feed_dict={
                                X: bat_X,
                                C: bat_C,
                                mle_index:(np.vstack((np.arange(bat_size), np.zeros(bat_size)+n)).T).astype(int).tolist(),
                                batch_size: bat_size
        })

        tri_mat = upp_tri_mat(np.zeros((n+1, n+1)))
        if bat_C[0] == 1:
            labels = np.zeros((bat_size,n+1))
        else:
            labels = np.ones((bat_size,n+1))

        _, _supervised_loss = session.run([train_supervised_op, supervised_loss], feed_dict={
                                X: bat_X,
                                batch_size: bat_size,
                                mask:tri_mat,
                                y_true:labels
        })

    if (n_epoch+1)%5==0 or n_epoch==num_epoches-1:

        unc_cen_gt_valid = list()
        Survival_valid = list()
        for n in range(time_steps):
            if n == 0:
                continue
            bat_X_valid = X_valid[T_valid==(n+1)].tolist()
            tri_mat_valid = upp_tri_mat(np.zeros((n+1, n+1)))
            bat_size_valid = np.sum(T_valid==(n+1)).astype(int)
            if bat_size_valid == 0:
                continue
            if (n+1) != time_steps:
                unc_cen_gt_valid.extend(np.ones(bat_size_valid).tolist())
            else:
                unc_cen_gt_valid.extend(np.zeros(bat_size_valid).tolist())
            _survivals_valid = session.run(survivals,
                                     feed_dict={X:bat_X_valid,
                                                batch_size:bat_size_valid,
                                                mask: tri_mat_valid})

            # _survivals_valid = np.array(_survivals_valid)
            # _survivals_valid = _survivals_valid.reshape(_survivals_valid.shape[1], _survivals_valid.shape[2])
            Survival_valid.extend(_survivals_valid)

        unc_cen_gt_valid = np.array(unc_cen_gt_valid)
        thrld_score = dict()
        for sur_thrld_valid in np.arange(0.1, 1.0, 0.02):
            unc_cen_valid = list()
            for ss_valid in Survival_valid:
                event_flag = False
                for s_valid in ss_valid:
                    if s_valid <= sur_thrld_valid:
                        event_flag = True
                        break
                if event_flag:
                    unc_cen_valid.append(1)
                else:
                    unc_cen_valid.append(0)
            unc_cen_valid = np.array(unc_cen_valid)
            unc_cen_acc_valid = accuracy_score(unc_cen_gt_valid, unc_cen_valid)
            unc_det_acc_valid = np.sum(unc_cen_valid[unc_cen_gt_valid==1])/float(np.sum(np.array(unc_cen_valid)==1))
            thrld_score[sur_thrld_valid] = 0.5*unc_cen_acc_valid+0.5*unc_det_acc_valid

        for thrld, score in thrld_score.items():
            if score > best_validation_accuracy:
                best_validation_accuracy = score
                best_thrld = thrld
                last_improvement = n_epoch+1
                saver.save(sess=session, save_path=save_path)
                print "**** ", n_epoch + 1, best_validation_accuracy, best_thrld

    if (n_epoch+1)-last_improvement > require_improvement:
        break

    # print "epoch: ", n_epoch, _mle_loss, _supervised_loss
    # print "epoch: ", n_epoch, _mle_loss
    print "epoch: ", n_epoch, _supervised_loss



saver.restore(sess=session, save_path=save_path)

unc_cen_gt = list()
Survival = list()
for n in range(time_steps):
    if n == 0:
        continue
    bat_X = X_test[T_test==(n+1)].tolist()
    tri_mat = upp_tri_mat(np.zeros((n+1, n+1)))
    bat_size = np.sum(T_test == (n+1)).astype(int)
    if bat_size == 0:
        continue
    if (n+1) != time_steps:
        unc_cen_gt.extend(np.ones(bat_size).tolist())
    else:
        unc_cen_gt.extend(np.zeros(bat_size).tolist())
    _survivals = session.run(survivals,
                               feed_dict={X:bat_X,
                                          batch_size: bat_size,
                                          mask: tri_mat})

    # _survivals = np.array(_survivals)
    # _survivals = _survivals.reshape(_survivals.shape[1],_survivals.shape[2])
    Survival.extend(_survivals)
unc_cen_gt = np.array(unc_cen_gt)
unc_cen = list()
for ss in Survival:
    event_flag = False
    for s in ss:
        if s <= best_thrld:
            event_flag = True
            break
    if event_flag:
        unc_cen.append(1)
    else:
        unc_cen.append(0)


unc_cen = np.array(unc_cen)
# C_test = C_test[T_test != 1]
unc_cen_acc = accuracy_score(unc_cen_gt, unc_cen)
# print np.sum(unc_cen[C_test == 1]), np.sum(np.array(unc_cen)==1)
print
print
print
unc_det_acc = np.sum(unc_cen[unc_cen_gt == 1])/float(np.sum(np.array(unc_cen)==1))
print "censor or uncensor ? : ", unc_cen_acc
print "given uncensor above, the true uncensor accuracy: ", unc_det_acc


exit(0)