import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
import os
sys.path.append("../")
from safdKit import upp_tri_mat,minibatch, prec_reca_F1, get_first_beat, early_det

global best_validation_accuracy
global best_thrld
global last_improvement
global require_improvement
from collections import defaultdict

# load data
#
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



# source_path = "../twitter/"
# X_train = np.load(source_path + "X_train_var.npy")
# T_train = np.load(source_path + "T_train_var.npy")
# C_train = np.load(source_path + "C_train_var.npy")
#
# X_test = np.load(source_path + "X_test_var.npy")
# T_test = np.load(source_path + "T_test_var.npy")
# C_test = np.load(source_path + "C_test_var.npy")
#
# X_valid = np.load(source_path + "X_valid_var.npy")
# T_valid = np.load(source_path + "T_valid_var.npy")
# C_valid = np.load(source_path + "C_valid_var.npy")
#
# n_input = 5
# print np.sum(T_train == 14), np.sum(T_train == 16)
# print np.sum(T_test == 14), np.sum(T_test == 16)
# print np.sum(T_valid == 14), np.sum(T_valid == 16)

# exit(0)





# print X_train.shape, T_train.shape, C_train.shape
# print X_test.shape, T_test.shape, C_test.shape
# print X_valid.shape, T_valid.shape, C_valid.shape
#
# exit(0)



# parameters setting
n_classes = 2
before_steps = 10

num_units = 32
learning_rate = .001

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)


X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
C = tf.placeholder(tf.float32, shape=[None,], name='C')
mle_index = tf.placeholder(tf.int32, shape=[None,2], name='mle_time')

y_v = tf.placeholder(tf.float32, shape=[None,None,2], name='y_v')

y_t = y_v[:,:,0]
y_t_1 = y_v[:,:,1]

batch_size = tf.placeholder(tf.int32, name='batch_size')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

alph_beta = tf.reshape(
                    tf.nn.softplus(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
                    [batch_size, -1,n_classes])

alpha = alph_beta[:,:,0]
beta = alph_beta[:,:,1]

lambdas = tf.subtract(tf.pow(tf.div(y_t_1,alpha),beta),
                                tf.pow(tf.div(y_t,alpha),beta))

mle_loss = tf.reduce_mean(
    tf.subtract(
                tf.reduce_sum(lambdas, axis=1),
                tf.multiply(tf.log(tf.subtract(tf.exp(tf.gather_nd(lambdas, mle_index)), 1)) ,C)
                # tf.multiply(tf.log(tf.subtract(tf.exp(tf.reduce_sum(lambdas, axis=1)), 1)), C)
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

mini_batch_size = 16
mini_batch_size_test_valid = 5
batch_x_train, batch_y_train, batch_t_train, batch_c_train = minibatch(X_train, T_train, C_train, mini_batch_size, n_input)
batch_x_valid, batch_y_valid, batch_t_valid, batch_c_valid = minibatch(X_valid, T_valid, C_valid, mini_batch_size_test_valid, n_input)
batch_x_test, batch_y_test, batch_t_test, batch_c_test = minibatch(X_test, T_test, C_test, mini_batch_size_test_valid, n_input)


# train_num = len(batch_x_train)*mini_batch_size
valid_num = len(batch_x_valid)*mini_batch_size_test_valid
test_num = len(batch_x_test)*mini_batch_size_test_valid

best_validation_accuracy = 0.0
best_thrld = 0.0
last_improvement = 0
require_improvement = 20
num_epoches = 500

session = tf.Session()
session.run(tf.global_variables_initializer())
tt = tf.trainable_variables()

# XX = session.run(tf.trainable_variables())

# print type(XX)
# print "----------------------------------------"
# print len(XX)

# for e in XX:
#     print np.asarray(e).shape
#     print
#     print
#
#
# exit(0)

def gen_y_t(v_t):
    coll = []
    for t in v_t:
        a = np.arange(t)[::-1].astype(float)
        b = np.arange(t)[::-1].astype(float) + 1
        coll.append(zip(a,b))
    return np.array(coll, dtype=np.float32)


for n_epoch in range(num_epoches):

    for batch_x, batch_t, batch_c in zip(batch_x_train, batch_t_train, batch_c_train):

        # y_t_y_t_1_v = gen_y_t(batch_t)
        bat_size = batch_x.shape[0]
        _, _mle_loss,  _lambdas = session.run([train_mle_op, mle_loss,lambdas],feed_dict={
                                X: batch_x,
                                C: batch_c,
                                mle_index:(np.vstack((np.arange(bat_size), batch_t-1)).T).astype(int).tolist(),
                                y_v:gen_y_t(batch_t),
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
            yy = []
            pp = []
            for batch_x, batch_y,  batch_t in zip(batch_x_valid, batch_y_valid, batch_t_valid):

                # y_t_y_t_1_v_valid = gen_y_t(batch_t)
                bat_size = batch_x.shape[0]
                seq_len = batch_x.shape[1]
                tri_mat = upp_tri_mat(np.zeros((seq_len, seq_len)))
                _pred_y_valid, _seq_pred_y = session.run([last_survival, survivals],feed_dict={X:batch_x,
                                                                                              batch_size:bat_size,
                                                                                              mask: tri_mat,
                                                                                              y_v:gen_y_t(batch_t)})
                pred_y_valid = np.zeros(_pred_y_valid.shape[0])
                pred_y_valid[np.where(_pred_y_valid>sur_thrld_valid)] = 1
                correct += np.sum(pred_y_valid == batch_y)

                # _seq_pred_y = _seq_pred_y[:,:before_steps]
                seq_pred_y_valid = np.zeros((_pred_y_valid.shape[0], before_steps))
                seq_pred_y_valid[np.where(_seq_pred_y[:,:before_steps]>sur_thrld_valid)] = 1
                batch_y = np.asarray([batch_y,]*before_steps).transpose()
                early_correct.append(np.sum(seq_pred_y_valid==batch_y, axis=0))
                yy.extend(batch_y)
                pp.extend(seq_pred_y_valid)

            yy = np.asarray(yy)
            pp = np.asarray(pp)
            gt = np.logical_not(yy).astype(float)
            pr = np.logical_not(pp).astype(float)
            _precision, _recall, _F1 = prec_reca_F1(gt, pr)

            last_corr_rate = correct/float(valid_num)
            seq_corr_rate = (np.sum(np.asarray(early_correct), axis=0)/float(valid_num))
            thrld_score[sur_thrld_valid] = np.mean(seq_corr_rate)

        for thrld, score in thrld_score.items():
            if score > best_validation_accuracy:
                best_validation_accuracy = score
                best_thrld = thrld
                last_improvement = n_epoch+1
                saver.save(sess=session, save_path=save_path)
                print "**** ", n_epoch + 1, best_validation_accuracy, best_thrld

    if (n_epoch+1)-last_improvement > require_improvement:
        break

    print "epoch: ", n_epoch, _mle_loss

saver.restore(sess=session, save_path=save_path)
correct = 0
early_correct = []
yy = []
pp = []


# print "best_threshold: ", best_thrld
#
# exit(0)
early_detect_steps = defaultdict(list)
# early_detect_rate = defaultdict(list)
# early_detect_num = defaultdict(list)
before_steps2 = 5
for batch_x, batch_y, batch_t in zip(batch_x_test, batch_y_test, batch_t_test):

    bat_size = batch_x.shape[0]
    seq_len = batch_x.shape[1]
    tri_mat = upp_tri_mat(np.zeros((seq_len, seq_len)))
    _pred_y_test, _seq_pred_y = session.run([last_survival, survivals],
                                            feed_dict={X:batch_x,
                                                       batch_size:bat_size,
                                                       mask:tri_mat,
                                                       y_v:gen_y_t(batch_t)
                                                       })
    pred_y_test = np.zeros(_pred_y_test.shape[0])
    pred_y_test[np.where(_pred_y_test > best_thrld)] = 1
    correct += np.sum(pred_y_test == batch_y)

    seq_events = _seq_pred_y[batch_y==0]
    if seq_events.shape[0] != 0:
        seq_pred_me_test = np.zeros((seq_events.shape[0], int(batch_t[0])))
        seq_pred_me_test[np.where(seq_events > best_thrld)] = 1
        # batch_msa = np.asarray([batch_y.flatten(), ]*int(batch_t[0])).transpose()
        batch_me = np.asarray([np.zeros(seq_events.shape[0]), ] * int(batch_t[0])).transpose()
        # fb = get_first_beat(seq_pred_me_test, batch_me)
        fb = early_det(seq_pred_me_test, batch_me)
        early_detect_steps[int(batch_t[0])].extend(np.multiply(np.ones(fb.shape[0]), int(batch_t[0]))-fb)

    # early_detect_rate[int(batch_t[0])].append(np.divide(fb.shape[0], batch_msa.shape[0], dtype="float"))
    # early_detect_num[int(batch_t[0])].append(fb.shape[0])

    # _seq_pred_y = _seq_pred_y[:, :before_steps]
    seq_pred_y_test = np.zeros((_pred_y_test.shape[0], before_steps2))
    seq_pred_y_test[np.where(_seq_pred_y[:, :before_steps2] > best_thrld)] = 1
    batch_y = np.asarray([batch_y, ] * before_steps2).transpose()
    early_correct.append(np.sum(seq_pred_y_test == batch_y, axis=0))
    # print "predicted: ", seq_pred_y_test
    # print "groud truth: ", batch_y
    # print "-----------------------------"
    yy.extend(batch_y)
    pp.extend(seq_pred_y_test)


yy = np.asarray(yy)
pp = np.asarray(pp)
gt = np.logical_not(yy).astype(float)
pr = np.logical_not(pp).astype(float)
_precision, _recall, _F1 = prec_reca_F1(gt, pr)

seq_corr_rate = (np.sum(np.asarray(early_correct), axis=0)/float(test_num))

print
print
print
print "precision: ", _precision
print "recall: ", _recall
print "F1: ", _F1
print "accuracy: ", seq_corr_rate

# print "------------------------------------------------------"
# for k, v in early_detect_steps.items():
#     print k, ": ", np.mean(v), len(v)

exit(0)