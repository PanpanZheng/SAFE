import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
import os
sys.path.append("../")

from safdKit import ran_seed, concordance_index, acc_pair_wtte, in_interval,pick_up_pair, acc_pair, cut_seq, cut_seq_last, cut_seq_0, cut_seq_mean, lambda2Survival
from sklearn.metrics import classification_report, accuracy_score



# parameters setting
n_input = 5
time_steps = 23
n_classes = 1

num_units = 32
learning_rate = .001

batch_size = 128
sigma = 2
theta = 0.5
sur_thrld = float(sys.argv[1])
# print type(sur_thrld),sur_thrld
# exit(0)

# convert lstm units to class probability.
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]),trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]),trainable=True)
para_list = [out_weights, out_bias]

# input placeholder
X = tf.placeholder(tf.float32, [None, None, n_input], name='X')
X_step = tf.placeholder(tf.float32, shape=(None,), name='X_time')
C = tf.placeholder(tf.float32, shape=(batch_size,), name='C')
mle_mask = tf.placeholder(tf.float32, [batch_size, time_steps], name='X_mask')
mle_index = tf.placeholder(tf.int32, shape=(batch_size,2), name='mle_time')

gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

lambdas = tf.reshape(
                    tf.nn.softplus(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
                    [-1,time_steps])
mle_loss = tf.reduce_mean(
    tf.subtract(
                tf.reduce_sum(tf.multiply(lambdas, mle_mask), axis=1),  
                tf.multiply(tf.log(tf.subtract(tf.exp(tf.gather_nd(lambdas, mle_index)), 1)) ,C)
                # tf.multiply(tf.log(tf.subtract(tf.exp(tf.reduce_sum(tf.multiply(lambdas, mle_mask), axis=1)),1)),C)
    )
)


rank_i_index = tf.placeholder(tf.int32, shape=(None, 2), name='i_index')
rank_j_index = tf.placeholder(tf.int32, shape=(None, 2), name='j_index')

rank_loss = tf.reduce_mean(
                tf.exp(
                    tf.divide(
                        tf.subtract(
                            tf.gather_nd(lambdas, rank_j_index),
                            tf.gather_nd(lambdas, rank_i_index)
                            )
                        ,sigma)
                    )
                )

#for mle
train_mle = tf.train.AdamOptimizer(learning_rate)
m_gvs = train_mle.compute_gradients(mle_loss)
m_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in m_gvs]
train_mle_op = train_mle.apply_gradients(m_capped_gvs)


#for rank
train_rank = tf.train.AdamOptimizer(learning_rate)
r_gvs = train_rank.compute_gradients(rank_loss)
r_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in r_gvs]
train_rank_op = train_rank.apply_gradients(r_capped_gvs)


#for check points.
saver = tf.train.Saver()
save_dir = '../checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "best_validation")

# load data
dest_path = "../Data2/"



# X_train = np.load(dest_path + "X_train_mean.npy")
# X_valid = np.load(dest_path + "X_valid_mean.npy")
# X_test = np.load(dest_path + "X_test_mean.npy")

X_train = np.load(dest_path + "X_train.npy")
T_train = np.load(dest_path + "T_train.npy")
C_train = np.load(dest_path + "C_train.npy")

X_valid = np.load(dest_path + "X_valid.npy")
T_valid = np.load(dest_path + "T_valid.npy")
C_valid = np.load(dest_path + "C_valid.npy")
Sur_Time_valid = T_valid + 1

X_test = np.load(dest_path + "X_test.npy")
T_test = np.load(dest_path + "T_test.npy")
C_test = np.load(dest_path + "C_test.npy")
Sur_Time_test = T_test + 1

# print X_train.shape, X_valid.shape, X_test.shape
# print Sur_Time_valid.shape, Sur_Time_test.shape
# exit(0)


# print X_train.shape, T_train.shape, C_train.shape
# print X_valid.shape, T_valid.shape, C_valid.shape
# print X_test.shape, T_test.shape, C_test.shape
#
# exit(0)

X_mask = np.zeros([X_train.shape[0], X_train.shape[1]])
for v, t in zip(X_mask, T_train):
    for i in range(t+1):
    # for i in range(t):
        v[i] = 1

acc_pair_train, usr2T_train = acc_pair(T_train,C_train)
acc_pair_test, usr2T_test = acc_pair(T_test,C_test)

n_batch = int(np.divide(len(X_train), batch_size))
ds = n_batch*batch_size
X_train, C_train, T_train, X_mask = X_train[:ds], C_train[:ds], T_train[:ds], X_mask[:ds]


session = tf.Session()
session.run(tf.global_variables_initializer())

global best_validation_accuracy
global last_improvement
global require_improvement

best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 10

num_epoches = 100
# sur_thrld = 0.7

for n_epoch in range(num_epoches):

    print "epoch: %s"%n_epoch

    for n in range(n_batch):

        _, _loss_mle, _lambdas_mle = session.run([train_mle_op, mle_loss, lambdas],feed_dict={
                                X: X_train[n * batch_size:(n + 1) * batch_size],
                                C: C_train[n * batch_size:(n + 1) * batch_size],
                                mle_mask: X_mask[n * batch_size:(n + 1) * batch_size],
                                mle_index: np.vstack((np.arange(batch_size), T_train[n * batch_size:(n + 1) * batch_size])).T
        })


        _X_pair = pick_up_pair([n*batch_size,(n+1)*batch_size], acc_pair_train)
        if len(_X_pair) == 0:
            continue
        X_pair = [[p[0]%batch_size, p[1]%batch_size] for p in _X_pair]
        i_index, j_index= list(), list()
        for x_p, _x_p in zip(X_pair, _X_pair):
            i_index.append([x_p[0],usr2T_train[_x_p[0]]])
            j_index.append([x_p[1],usr2T_train[_x_p[0]]])

        _, _loss_rank, _lambdas_rank = session.run([train_rank_op, rank_loss, lambdas],feed_dict={
                                X: X_train[n*batch_size:(n+1)*batch_size],
                                rank_i_index:i_index,
                                rank_j_index:j_index
        })


    if (n_epoch+1)%5 == 0 or n_epoch == num_epoches-1:

        _lambdas_valid = np.array(
            session.run([lambdas], feed_dict={X:X_valid})
        )
        _lambdas_valid = _lambdas_valid.reshape(_lambdas_valid.shape[1], _lambdas_valid.shape[2])
        survivals_valid = list()
        for H in _lambdas_valid:
            survivals_valid.append(lambda2Survival(H))

        censor_detect = list()
        suspended_candidate_survival = list()
        suspended_candidate_survival_time = list()
        for j, survival in enumerate(survivals_valid):
            flag = True
            for i in range(time_steps-1):
                if survival[i] <= sur_thrld:
                    censor_detect.append(i+1)
                    suspended_candidate_survival.append(survival)
                    suspended_candidate_survival_time.append(Sur_Time_valid[j])
                    flag = False
                    break
            if flag:
                censor_detect.append(time_steps)
        censor_detected_rate = np.sum((np.array(censor_detect)==23)==(C_valid==0))/float(np.array(censor_detect).shape[0])
        print "**********************"
        for a, b in zip(Sur_Time_valid, survivals_valid):
            print a, ": ", b
        print "**********************"

        # print "**********************"
        # for s, d, c in zip(Sur_Time_valid, censor_detect, C_valid):
        #     print s, d , c
        # print "*********************"
        # exit(0)

        sus_detect = list()
        count_censor = 0
        for j, survival in enumerate(suspended_candidate_survival):
            if suspended_candidate_survival_time[j] == 23:
                count_censor += 1
                continue
            flag = True
            for i in range(suspended_candidate_survival_time[j]):
                if survival[i] <= sur_thrld:
                    sus_detect.append(i+1)
                    flag = False
                    break
            if flag:
                sus_detect.append(-1)

        suspend_detected_rate = np.sum(np.array(sus_detect)!=-1)/float(np.array(suspended_candidate_survival).shape[0])
        # print "---------", count_censor, suspend_detected_rate, np.sum(np.array(sus_detect)!=-1), float(np.array(suspended_candidate_survival).shape[0])
        acc_valid = .5*censor_detected_rate + .5*suspend_detected_rate
        print censor_detected_rate, suspend_detected_rate

        if acc_valid > best_validation_accuracy:

            best_validation_accuracy = acc_valid
            last_improvement = n_epoch+1
            saver.save(sess=session, save_path=save_path)
            print "**** ", n_epoch+1, best_validation_accuracy
        else:
            print "#### ", n_epoch+1, acc_valid

    if (n_epoch+1)-last_improvement > require_improvement:
        break



saver.restore(sess=session, save_path=save_path)

_lambdas_test = np.array(
    session.run([lambdas], feed_dict={X:X_test})
)
_lambdas_test = _lambdas_test.reshape(_lambdas_test.shape[1], _lambdas_test.shape[2])

survivals_test = list()
for H in _lambdas_test:
    survivals_test.append(lambda2Survival(H))

censor_detect_test = list()
suspended_candidate_survival_test = list()
suspended_candidate_survival_time_test = list()
for j, survival in enumerate(survivals_test):
    flag = True
    for i in range(time_steps-1):
        if survival[i] <= sur_thrld:
            censor_detect_test.append(i+1)
            suspended_candidate_survival_test.append(survival)
            suspended_candidate_survival_time_test.append(Sur_Time_test[j])
            flag = False
            break
    if flag:
        censor_detect_test.append(time_steps)

censor_detected_rate_test = np.sum((np.array(censor_detect_test)==23)==(C_test==0))/float(np.array(censor_detect_test).shape[0])

sus_detect_test = list()
for j, survival in enumerate(suspended_candidate_survival_test):
    if suspended_candidate_survival_time_test[j] == 23:
        continue
    flag = True
    for i in range(suspended_candidate_survival_time_test[j]):
        if survival[i] <= sur_thrld:
            sus_detect_test.append(i+1)
            flag = False
            break
    if flag:
        sus_detect_test.append(-1)

acc_valid_test = np.sum(np.array(sus_detect_test)!=-1)/float(np.array(sus_detect_test).shape[0])

print "censor_detect_rate: ", censor_detected_rate_test
print "suspend_detect_rate: ", acc_valid_test

# for k in np.arange(2):
#
#     _lambdas_test = np.array(
#         session.run([lambdas], feed_dict={X:X_test})
#     )
#     _lambdas_test = _lambdas_test.reshape(_lambdas_test.shape[1], _lambdas_test.shape[2])
#
#     survivals_test = list()
#     for H in _lambdas_test:
#         survivals_test.append(lambda2Survival(H))
#
#     T_test_pred = list()
#     for survival in survivals_test:
#         flag = True
#         for i in range(time_steps-1):
#             if survival[i] <= sur_thrld:
#                 T_test_pred.append(i + 1)
#                 flag = False
#                 break
#         if flag:
#             T_test_pred.append(time_steps - 1)
#     acc_test = np.sum((np.array(T_test_pred) == 22) == (C_test == 0)) / float(np.array(T_test_pred).shape[0])
#     # acc_test = accuracy_score(T_test, np.array(T_test_pred))
#     print "test %s: "%k, acc_test



# acc_valid = 1 - not_detected_rate
# sus_detect = list()
# for j, survival in enumerate(survivals_valid):
#
#     flag = True
#     for i in range(Sur_Time_valid[j]):
#         if survival[i] <= sur_thrld:
#             sus_detect.append(i+1)
#             flag = False
#             break
#     if flag:
#         sus_detect.append(-1)
#
# not_detected_rate = np.sum(np.array(sus_detect)==-1)/float(np.array(sus_detect).shape[0])
# acc_valid = 1-not_detected_rate

# print
# print "Undetected: "
# for sur, pred_sur in zip(Sur_Time_valid[np.array(T_valid_pred) == -1], np.array(survivals_valid)[np.array(T_valid_pred) == -1]):
#     print sur, pred_sur
# print
# print
# print "Detected: "
# for sur, pred_sur in zip(Sur_Time_valid[np.array(T_valid_pred) != -1], np.array(survivals_valid)[np.array(T_valid_pred) != -1]):
#     print sur, pred_sur
# exit(0)



# T_test_pred = list()
# for j, survival in enumerate(survivals_test):
#     flag = True
#     for i in range(Sur_Time_test[j]):
#         if survival[i] <= sur_thrld:
#             T_test_pred.append(i+1)
#             flag = False
#             break
#     if flag:
#         T_test_pred.append(-1)
# 
# 
# not_detect_rate = np.sum(np.array(T_test_pred) == -1)/float(np.array(T_test_pred).shape[0])
# acc_valid = 1 - not_detect_rate
# 
# print "test: ", acc_valid
#


#
# T_pred_test=[]
# H = np.array(
#     session.run([lambdas], feed_dict={X:X_test})
# )
# H = H.reshape(H.shape[1],H.shape[2])
# for hs in H:
#     T_pred_test.append(np.argmax(hs)+1)
#
# mae = np.mean(np.abs(T_test-T_pred_test))
# T_pred_test = np.array(T_pred_test)
#
# count = 0
# for p in acc_pair_test:
#     if T_pred_test[p[0]] < T_pred_test[p[1]]:
#         count += 1
# CI = count/float(len(acc_pair_test))
# print("epoch %s: %s %s %s %s" %(n_epoch, np.mean(rank_batch_loss), np.mean(mle_batch_loss), mae, CI))
