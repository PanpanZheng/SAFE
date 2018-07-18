import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
import os
sys.path.append("../")
from safdKit import upp_tri_mat, ran_seed, minibatch
from sklearn.metrics import classification_report, accuracy_score

global best_validation_accuracy
global best_thrld
global last_improvement
global require_improvement

# parameters setting
n_input = 5
time_steps = 21
n_classes = 1

num_units = 32
learning_rate = .001

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
                # tf.multiply(tf.log(tf.subtract(tf.exp(tf.gather_nd(lambdas, mle_index)), 1)) ,C),
                tf.multiply(tf.log(tf.subtract(tf.exp(tf.reduce_sum(lambdas, axis=1)), 1)+tf.exp(-20.)), C)
    )
)

mask = tf.placeholder(tf.float32, shape=[None,None], name='upper_matrix')  # upper triangle matrix
survivals = tf.exp(-tf.matmul(lambdas, mask))
last_survival = survivals[:,-1]


#for mle
train_mle = tf.train.AdamOptimizer(learning_rate)
m_gvs = train_mle.compute_gradients(mle_loss)
m_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in m_gvs]
train_mle_op = train_mle.apply_gradients(m_capped_gvs)


#for supervised
# y_true = tf.placeholder(tf.int32, shape=[None, None], name="y_real")
# supervised_loss = tf.losses.mean_squared_error(y_true, survivals)
#
# train_supervised = tf.train.AdamOptimizer(learning_rate)
# s_gvs = train_supervised.compute_gradients(supervised_loss)
# s_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in s_gvs]
# train_supervised_op = train_supervised.apply_gradients(s_capped_gvs)


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

X_valid = np.load(source_path + "X_valid_var.npy")
T_valid = np.load(source_path + "T_valid_var.npy")
C_valid = np.load(source_path + "C_valid_var.npy")

X_test = np.load(source_path + "X_test_var.npy")
T_test = np.load(source_path + "T_test_var.npy")
C_test = np.load(source_path + "C_test_var.npy")

mini_batch_size = 16
batch_x_train, batch_y_train, batch_t_train, batch_c_train = minibatch(X_train, T_train, C_train, mini_batch_size, n_input)
batch_x_valid, batch_y_valid, batch_t_valid, batch_c_valid = minibatch(X_valid, T_valid, C_valid, mini_batch_size, n_input)
batch_x_test, batch_y_test, batch_t_test, batch_c_test = minibatch(X_test, T_test, C_test, mini_batch_size, n_input)

# train_num = len(batch_x_train)*mini_batch_size
valid_num = len(batch_x_valid)*mini_batch_size
test_num = len(batch_x_test)*mini_batch_size

best_validation_accuracy = 0.0
best_thrld = 0.0
last_improvement = 0
require_improvement = 20
num_epoches = 500

session = tf.Session()
session.run(tf.global_variables_initializer())

for n_epoch in range(num_epoches):

    for batch_x, batch_t, batch_c in zip(batch_x_train, batch_t_train, batch_c_train):

        bat_size = batch_x.shape[0]
        _, _mle_loss = session.run([train_mle_op, mle_loss],feed_dict={
                                X: batch_x,
                                C: batch_c,
                                mle_index:(np.vstack((np.arange(bat_size), batch_t-1)).T).astype(int).tolist(),
                                batch_size:bat_size
        })

        # seq_len = batch_x.shape[1]
        # tri_mat = upp_tri_mat(np.zeros((seq_len, seq_len)))
        # if batch_c[0] == 1:
        #     labels = np.zeros((bat_size,seq_len))
        # else:
        #     labels = np.ones((bat_size,seq_len))
        #
        # _, _supervised_loss = session.run([train_supervised_op, supervised_loss], feed_dict={
        #                         X: batch_x,
        #                         batch_size: bat_size,
        #                         mask:tri_mat,
        #                         y_true:labels
        # })


    if (n_epoch+1)%5==0 or n_epoch==num_epoches-1:

        thrld_score = dict()
        for sur_thrld_valid in np.arange(0.1, 1.0, 0.02):
            correct = 0
            early_correct = []
            for batch_x, batch_y in zip(batch_x_valid, batch_y_valid):
                bat_size = batch_x.shape[0]
                seq_len = batch_x.shape[1]
                tri_mat = upp_tri_mat(np.zeros((seq_len, seq_len)))
                _pred_y_valid, _seq_pred_y = session.run([last_survival, survivals],feed_dict={X:batch_x,
                                                                                              batch_size:bat_size,
                                                                                              mask: tri_mat})
                pred_y_valid = np.zeros(_pred_y_valid.shape[0])
                pred_y_valid[np.where(_pred_y_valid>sur_thrld_valid)] = 1
                correct += np.sum(pred_y_valid == batch_y)

                _seq_pred_y = _seq_pred_y[:,-10:]
                seq_pred_y_valid = np.zeros((_pred_y_valid.shape[0], 10))
                seq_pred_y_valid[np.where(_seq_pred_y>sur_thrld_valid)] = 1
                batch_y = np.asarray([batch_y,]*10).transpose()
                early_correct.append(np.sum(seq_pred_y_valid==batch_y, axis=0))


            last_corr_rate = correct/float(valid_num)
            seq_corr_rate = (np.sum(np.asarray(early_correct), axis=0)/float(valid_num))
            thrld_score[sur_thrld_valid] = np.mean(seq_corr_rate)

        for thrld, score in thrld_score.items():
            if score > best_validation_accuracy:
                best_validation_accuracy = score
                best_thrld = thrld
                last_improvement = n_epoch+1
                saver.save(sess=session, save_path=save_path)
                # print "**** ", n_epoch + 1, best_validation_accuracy, best_thrld

    if (n_epoch+1)-last_improvement > require_improvement:
        break

    # print "epoch: ", n_epoch, _mle_loss

saver.restore(sess=session, save_path=save_path)
correct = 0
early_correct = []
for batch_x, batch_y in zip(batch_x_test, batch_y_test):

    bat_size = batch_x.shape[0]
    seq_len = batch_x.shape[1]
    tri_mat = upp_tri_mat(np.zeros((seq_len, seq_len)))
    _pred_y_test, _seq_pred_y = session.run([last_survival, survivals],
                                            feed_dict={X:batch_x,
                                                       batch_size:bat_size,
                                                       mask:tri_mat
                                                       })
    pred_y_test = np.zeros(_pred_y_test.shape[0])
    pred_y_test[np.where(_pred_y_test > best_thrld)] = 1
    correct += np.sum(pred_y_test == batch_y)

    _seq_pred_y = _seq_pred_y[:, -10:]
    seq_pred_y_test = np.zeros((_pred_y_test.shape[0], 10))
    seq_pred_y_test[np.where(_seq_pred_y > best_thrld)] = 1
    batch_y = np.asarray([batch_y, ] * 10).transpose()
    early_correct.append(np.sum(seq_pred_y_test == batch_y, axis=0))

last_corr_rate = correct/float(test_num)
seq_corr_rate = (np.sum(np.asarray(early_correct), axis=0)/float(test_num))

print seq_corr_rate
print "threshold: ", best_thrld

exit(0)