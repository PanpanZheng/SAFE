import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
sys.path.append("../")

from safdKit import ran_seed, concordance_index, acc_pair_wtte, in_interval,pick_up_pair, acc_pair

# parameters setting
n_input = 5
time_steps = 23
n_classes = 1

num_units = 32
learning_rate = .001

batch_size = 128
sigma = 2
theta = 0.5

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
# outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, sequence_length=X_step, dtype="float32")

outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

lambdas = tf.reshape(
                    tf.nn.softplus(tf.matmul(tf.reshape(outputs,[-1,num_units]),out_weights)+ out_bias),
                    [-1,time_steps])
mle_loss = tf.reduce_mean(
    tf.subtract(
                tf.reduce_sum(tf.multiply(lambdas, mle_mask), axis=1),  
                tf.multiply(tf.log(tf.subtract(tf.exp(tf.gather_nd(lambdas, mle_index)), 1)) ,C)
    )
)


# rank_i_index = tf.placeholder(tf.int32, shape=(None, 2), name='i_index')
# rank_j_index = tf.placeholder(tf.int32, shape=(None, 2), name='j_index')


rank_i_index = tf.placeholder(tf.int32, shape=(5561, 2), name='i_index')
rank_j_index = tf.placeholder(tf.int32, shape=(5561, 2), name='j_index')

rank_loss = tf.reduce_mean(
                tf.exp(
                    tf.divide(
                        tf.subtract(
                            tf.gather_nd(lambdas, rank_j_index),  # hazard_series, whole, j_ind partial.
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



# load data
dest_path = "../Data/"

X_train = np.load(dest_path + "X_train.npy")
T_train = np.load(dest_path + "T_train.npy")
C_train = np.load(dest_path + "C_train.npy")

X_mask = np.zeros([X_train.shape[0], X_train.shape[1]])
for v, t in zip(X_mask, T_train):
    for i in range(t+1):
    # for i in range(t):
        v[i] = 1

acc_pair_train, usr2T_train = acc_pair(T_train,C_train)

X_test = np.load(dest_path + "X_test.npy")
T_test = np.load(dest_path + "T_test.npy")
C_test = np.load(dest_path + "C_test.npy")

X_event, T_event, X_censor, T_censor = [], [], [], []
for i, c in enumerate(C_test):
    if c == 1:
        X_event.append(X_test[i])
        T_event.append(T_test[i])
    else:
        X_censor.append(X_test[i])
        T_censor.append(T_test[i])
X_event, T_event, X_censor, T_censor = np.asarray(X_event), np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)

acc_pair_test, usr2T_test = acc_pair(T_event,np.ones(T_event.shape[0]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_batch = int(np.divide(len(X_train), batch_size))

ds = n_batch*batch_size
X_train, C_train, T_train, X_mask = X_train[:ds], C_train[:ds], T_train[:ds], X_mask[:ds]

for n_epoch in range(100):
    rank_batch_loss = []
    batch_mae = []
    for n in range(n_batch):

        # _, _loss_mle, _lambdas = sess.run([train_mle_op, mle_loss, lambdas],feed_dict={
        #                         X: X_train[n * batch_size:(n + 1) * batch_size],
        #                         C: C_train[n * batch_size:(n + 1) * batch_size],
        #                         X_step: T_train[n * batch_size:(n + 1) * batch_size],
        #                         mle_mask: X_mask[n * batch_size:(n + 1) * batch_size],
        #                         mle_index: np.vstack((np.arange(batch_size), T_train[n * batch_size:(n + 1) * batch_size])).T
        # })


        _X_pair = pick_up_pair([n*batch_size,(n+1)*batch_size], acc_pair_train)
        X_pair = [[p[0]%batch_size, p[1]%batch_size] for p in _X_pair]
        i_index, j_index= list(), list()
        for x_p, _x_p in zip(X_pair, _X_pair):
            i_index.append([x_p[0],usr2T_train[_x_p[0]]])
            j_index.append([x_p[1],usr2T_train[_x_p[0]]])

        i_index, j_index = np.array(i_index), np.array(j_index)
        # print
        # print i_index.shape, j_index.shape
        # exit(0)

        _, _loss_rank, _lambdas = sess.run([train_rank_op, rank_loss, lambdas],feed_dict={
                                X: X_train[n*batch_size:(n+1)*batch_size],
                                rank_i_index:i_index,
                                rank_j_index:j_index
        })

        rank_batch_loss.append(_loss_rank)
        T_pred_train = []
        for hs in _lambdas:
            T_pred_train.append(np.argmax(hs))
        mae = np.mean(np.abs(T_train[n * batch_size:(n + 1) * batch_size]-T_pred_train))
        batch_mae.append(mae)
        print np.mean(rank_batch_loss), np.mean(batch_mae)
        exit(0)

    print np.mean(rank_batch_loss), np.mean(batch_mae)


    # T_pred_test=[]
    # H = np.array(
    #     sess.run([lambdas], feed_dict={X:X_event, X_step:T_event})
    # )
    # H = H.reshape(H.shape[1],H.shape[2])
    # for hs in H:
    #     T_pred_test.append(np.argmax(hs))
    #
    #
    #
    # mae = np.mean(np.abs(T_event-T_pred_test))
    #
    # # print("epoch %s: %s %s %s" %(n_epoch, np.mean(mle_batch_loss), np.mean(batch_mae), mae))
    # # print T_pred_test[0:20]
    # # print T_event[0:20]
    # # print
    # # print
    #
    #
    # acc_pair_test, usr2T_test = acc_pair_wtte(T_event)
    # count = 0
    # for p in acc_pair_test:
    #     if T_pred_test[p[0]] < T_pred_test[p[1]]:
    #         count += 1
    # CI = count/float(len(acc_pair_test))
    # print("epoch %s: %s %s %s %s" %(n_epoch, np.mean(mle_batch_loss), np.mean(batch_mae), mae, CI))
