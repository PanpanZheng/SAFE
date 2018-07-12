import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
sys.path.append("../")
from sklearn.metrics import classification_report, accuracy_score

# parameters setting
n_input = 5
n_classes = 1

num_units = 32
learning_rate = .001

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)

X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
y_true = tf.placeholder(tf.int32, shape=[None,], name="y_real")
batch_size = tf.placeholder(tf.int32, name='batch_size')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

y_pred = tf.reshape(
            tf.nn.sigmoid(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
            [batch_size, -1])[:,-1]

mse_loss = tf.losses.mean_squared_error(y_true, y_pred)

#for supervised
train_supervised = tf.train.AdamOptimizer(learning_rate)
s_gvs = train_supervised.compute_gradients(mse_loss)
s_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in s_gvs]
train_supervised_op = train_supervised.apply_gradients(s_capped_gvs)


# load data
dest_path = "../Data3/"

X_train_12 = np.load(dest_path + "X_train_12.npy")
X_test_12 = np.load(dest_path + "X_test_12.npy")

X_train_17 = np.load(dest_path + "X_train_17.npy")
X_test_17 = np.load(dest_path + "X_test_17.npy")

X_train_18 = np.load(dest_path + "X_train_18.npy")
X_test_18 = np.load(dest_path + "X_test_18.npy")

X_train_19 = np.load(dest_path + "X_train_19.npy")
X_test_19 = np.load(dest_path + "X_test_19.npy")

X_train_21 = np.load(dest_path + "X_train_21.npy")
X_test_21 = np.load(dest_path + "X_test_21.npy")


session = tf.Session()
session.run(tf.global_variables_initializer())

num_epoches = 500

for n_epoch in range(num_epoches):

    for i in [12,17,18,19,21]:

        if i == 12:
            bat_X = X_train_12
        elif i == 17:
            bat_X = X_train_17
        elif i == 18:
            bat_X = X_train_18
        elif i == 19:
            bat_X = X_train_19
        elif i == 21:
            bat_X = X_train_21

        bat_size = bat_X.shape[0]

        if i != 21:
            bat_y = np.ones(bat_size)
        else:
            bat_y = np.zeros(bat_size)

        _, _mse_loss, _y_pred = session.run([train_supervised_op, mse_loss, y_pred],feed_dict={
                                X: bat_X,
                                y_true: bat_y,
                                batch_size: bat_size
        })

    print "epoch: ", n_epoch, _mse_loss

_y_pred_test = list()

for i in [12, 17, 18, 19, 21]:

    if i == 12:
        bat_X = X_test_12
    elif i == 17:
        bat_X = X_test_17
    elif i == 18:
        bat_X = X_test_18
    elif i == 19:
        bat_X = X_test_19
    elif i == 21:
        bat_X = X_test_21

    bat_size = bat_X.shape[0]

    _y_pred = session.run([y_pred], feed_dict={
        X: bat_X,
        batch_size: bat_size
    })

    _y_pred = np.array(_y_pred).reshape((bat_size,))
    _y_pred_test.extend(_y_pred)

_y_test = np.concatenate((np.ones(240),np.zeros(240)))
_y_pred_test = np.array(_y_pred_test)
_y_pred_test = (np.array(_y_pred_test)>0.5).astype(int)
acc = accuracy_score(_y_test, _y_pred_test)
print acc

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