import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import sys
sys.path.append("../")
from sklearn.metrics import classification_report, accuracy_score

# parameters setting
n_input = 5
n_classes = 1

num_units = 32
learning_rate = .001
time_steps = 21

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)

X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
y_true = tf.placeholder(tf.int32, shape=[None,], name="y_real")
batch_size = tf.placeholder(tf.int32, name='batch_size')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

y_seq_pred = tf.reshape(
            tf.nn.sigmoid(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
            [batch_size, -1])

y_pred = y_seq_pred[:,-1]

# y_pred = tf.reshape(
#             tf.nn.sigmoid(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
#             [batch_size, -1])[:,-1]

mse_loss = tf.losses.sigmoid_cross_entropy(y_true, y_pred)

#for supervised
train_supervised = tf.train.AdamOptimizer(learning_rate)
s_gvs = train_supervised.compute_gradients(mse_loss)
s_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in s_gvs]
train_supervised_op = train_supervised.apply_gradients(s_capped_gvs)


saver = tf.train.Saver()
save_dir = '../checkpoints_rnn/'
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
        if n != 21:
            bat_y = np.zeros(bat_size)
        else:
            bat_y = np.ones(bat_size)

        _, _mse_loss = session.run([train_supervised_op, mse_loss],feed_dict={
                                X: bat_X,
                                y_true: bat_y,
                                batch_size: bat_size
                                })

    if (n_epoch+1)%5==0 or n_epoch==num_epoches-1:

        unc_cen_gt_valid = list()
        Survival_valid = list()
        for n in range(time_steps):
            if n == 0:
                continue
            bat_X_valid = X_valid[T_valid==(n+1)].tolist()
            bat_size_valid = np.sum(T_valid==(n+1)).astype(int)
            if bat_size_valid == 0:
                continue

            if (n+1) != time_steps:
                unc_cen_gt_valid.extend(np.zeros(bat_size_valid).tolist())
            else:
                unc_cen_gt_valid.extend(np.ones(bat_size_valid).tolist())

            _survivals_valid = session.run([y_seq_pred], feed_dict={
                                     X: bat_X_valid,
                                     batch_size: bat_size_valid
                                    })

            _survivals_valid = np.array(_survivals_valid)
            _survivals_valid = _survivals_valid.reshape(_survivals_valid.shape[1], _survivals_valid.shape[2])
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
                    unc_cen_valid.append(0)
                else:
                    unc_cen_valid.append(1)
            unc_cen_valid = np.array(unc_cen_valid)
            unc_cen_acc_valid = accuracy_score(unc_cen_gt_valid, unc_cen_valid)
            unc_det_acc_valid = np.sum(unc_cen_valid[unc_cen_gt_valid==0]==0)/float(np.sum(np.array(unc_cen_valid)==0))
            thrld_score[sur_thrld_valid] = 0.5*unc_cen_acc_valid+0.5*unc_det_acc_valid

        for thrld, score in thrld_score.items():
            if score > best_validation_accuracy:
                best_validation_accuracy = score
                best_thrld = thrld
                last_improvement = n_epoch + 1
                saver.save(sess=session, save_path=save_path)
                print "**** ", n_epoch+1, best_validation_accuracy, best_thrld

    if (n_epoch+1)-last_improvement > require_improvement:
        break
    print "epoch: ", n_epoch, _mse_loss

saver.restore(sess=session, save_path=save_path)
unc_cen_gt_test = list()
Survival_test = list()
for n in range(time_steps):
    if n == 0:
        continue
    bat_X = X_test[T_test==(n+1)].tolist()
    bat_size = np.sum(T_test==(n+1)).astype(int)
    if bat_size == 0:
        continue

    if (n+1) != time_steps:
        unc_cen_gt_test.extend(np.zeros(bat_size).tolist())
    else:
        unc_cen_gt_test.extend(np.ones(bat_size).tolist())

    _survivals_test = session.run([y_seq_pred], feed_dict={
                                   X: bat_X,
                                   batch_size: bat_size
                                })

    _survivals_test = np.array(_survivals_test)
    _survivals_test = _survivals_test.reshape(_survivals_test.shape[1], _survivals_test.shape[2])
    Survival_test.extend(_survivals_test)

unc_cen_gt_test = np.array(unc_cen_gt_test)
unc_cen_test = list()
for ss_test in Survival_test:
    event_flag = False
    for s in ss_test:
        if s <= best_thrld:
            event_flag = True
            break
    if event_flag:
        unc_cen_test.append(0)
    else:
        unc_cen_test.append(1)

unc_cen_test = np.array(unc_cen_test)
unc_cen_acc_test = accuracy_score(unc_cen_gt_test, unc_cen_test)
unc_det_acc_test = np.sum((unc_cen_test[unc_cen_gt_test==0])==0)/float(np.sum(np.array(unc_cen_test)==0))


print
print
print
print "best threshold: %s"%best_thrld
print "censor or uncensor ? : ", unc_cen_acc_test
print "given uncensor above, the true uncensor accuracy: ", unc_det_acc_test


exit(0)