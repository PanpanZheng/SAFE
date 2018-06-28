import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
sys.path.append("../")

from safdKit import ran_seed, concordance_index, acc_pair_wtte, in_interval,pick_up_pair, acc_pair, cut_seq, cut_seq_last, cut_seq_0, cut_seq_mean

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


rank_i_index = tf.placeholder(tf.int32, shape=(None, 2), name='i_index')
rank_j_index = tf.placeholder(tf.int32, shape=(None, 2), name='j_index')


# rank_i_index = tf.placeholder(tf.int32, shape=(5561, 2), name='i_index')
# rank_j_index = tf.placeholder(tf.int32, shape=(5561, 2), name='j_index')

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

# pad_med = sys.argv[1]
# tra_ful_mixed = sys.argv[2]
# tes_ful_par = sys.argv[3]


pad_med = "0"
tra_ful_mixed = "full"
tes_ful_par = "partial"

if pad_med == "0":
    X_train = np.load(dest_path + "X_train_pad_0.npy")
    T_train = np.load(dest_path + "T_train_pad_0.npy")
    C_train = np.load(dest_path + "C_train_pad_0.npy")
elif pad_med == "last":
    X_train = np.load(dest_path + "X_train_pad_last.npy")
    T_train = np.load(dest_path + "T_train_pad_last.npy")
    C_train = np.load(dest_path + "C_train_pad_last.npy")
elif pad_med == "mean":
    X_train = np.load(dest_path + "X_train_pad_mean.npy")
    T_train = np.load(dest_path + "T_train_pad_mean.npy")
    C_train = np.load(dest_path + "C_train_pad_mean.npy")

if tra_ful_mixed == "mix":

    X_train_C = list()
    for x in X_train:
        X_train_C.append(x.tolist())

    X_train_C = np.array(X_train.tolist())
    X_train_C = cut_seq_last(X_train_C, T_train, 0.5)
    X_train_C = cut_seq_mean(X_train_C, T_train, 0.5)

    X_train = X_train.tolist() + X_train_C.tolist()
    T_train = T_train.tolist() + T_train.tolist()
    C_train = C_train.tolist() + C_train.tolist()


    X_train, T_train, C_train = np.array(X_train), np.array(T_train), np.array(C_train)

    s = ran_seed(X_train.shape[0])
    X_train, T_train, C_train = X_train[s], T_train[s], C_train[s]


X_mask = np.zeros([X_train.shape[0], X_train.shape[1]])
for v, t in zip(X_mask, T_train):
    for i in range(t+1):
    # for i in range(t):
        v[i] = 1

acc_pair_train, usr2T_train = acc_pair(T_train,C_train)


if pad_med == "0":
    X_test = np.load(dest_path + "X_test_pad_0.npy")
    T_test = np.load(dest_path + "T_test_pad_0.npy")
    C_test = np.load(dest_path + "C_test_pad_0.npy")
elif pad_med == "last":
    X_test = np.load(dest_path + "X_test_pad_last.npy")
    T_test = np.load(dest_path + "T_test_pad_last.npy")
    C_test = np.load(dest_path + "C_test_pad_last.npy")
elif pad_med == "mean":
    X_test = np.load(dest_path + "X_test_pad_mean.npy")
    T_test = np.load(dest_path + "T_test_pad_mean.npy")
    C_test = np.load(dest_path + "C_test_pad_mean.npy")


X_event, T_event, X_censor, T_censor = [], [], [], []
for i, c in enumerate(C_test):
    if c == 1:
        X_event.append(X_test[i].tolist())
        T_event.append(T_test[i].tolist())
    else:
        X_censor.append(X_test[i].tolist())
        T_censor.append(T_test[i].tolist())

if tes_ful_par == "full":
    X_event, T_event, X_censor, T_censor = np.asarray(X_event), np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)
elif pad_med == "0" and tes_ful_par == "partial":
    T_event, X_censor, T_censor = np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)
    X_event, dis_len, cut_points = cut_seq_0(X_event,T_event,n_input,0.5)
    # print dis_len
    # exit(0)
elif pad_med == "last" and tes_ful_par == "partial":
    T_event, X_censor, T_censor = np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)
    X_event, dis_len, cut_points = cut_seq_last(X_event, T_event, 0.5)
    # print dis_len
    # exit(0)
elif pad_med == "mean" and tes_ful_par == "partial":
    T_event, X_censor, T_censor = np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)
    X_event, dis_len, cut_points = cut_seq_mean(X_event, T_event, 0.5)
    # print dis_len
    # exit(0)

# T_event, X_censor, T_censor = np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)
# X_event = cut_seq_last_0(X_event,T_event,5, 0.5)

# T_event, X_censor, T_censor = np.asarray(T_event), np.asarray(X_censor), np.asarray(T_censor)
# X_event = cut_seq_last(X_event,T_event, 0.5)
# X_event = cut_seq_mean(X_event,T_event, 0.5)

acc_pair_test, usr2T_test = acc_pair(T_event,np.ones(T_event.shape[0]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_batch = int(np.divide(len(X_train), batch_size))

ds = n_batch*batch_size
X_train, C_train, T_train, X_mask = X_train[:ds], C_train[:ds], T_train[:ds], X_mask[:ds]


for n_epoch in range(30):
    rank_batch_loss = []
    mle_batch_loss = []
    # batch_mae = []
    for n in range(n_batch):

        _, _loss_mle, _lambdas_mle = sess.run([train_mle_op, mle_loss, lambdas],feed_dict={
                                X: X_train[n * batch_size:(n + 1) * batch_size],
                                C: C_train[n * batch_size:(n + 1) * batch_size],
                                # X_step: T_train[n * batch_size:(n + 1) * batch_size],
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

        _, _loss_rank, _lambdas_rank = sess.run([train_rank_op, rank_loss, lambdas],feed_dict={
                                X: X_train[n*batch_size:(n+1)*batch_size],
                                rank_i_index:i_index,
                                rank_j_index:j_index
        })

        rank_batch_loss.append(_loss_rank)
        mle_batch_loss.append(_loss_mle)
        # T_pred_train = []
        # for hs in _lambdas:
        #     T_pred_train.append(np.argmax(hs))
        # mae = np.mean(np.abs(T_train[n * batch_size:(n + 1) * batch_size]-T_pred_train))
        # batch_mae.append(mae)

    # print np.mean(rank_batch_loss), np.mean(batch_mae)

    T_pred_test=[]
    H = np.array(
        sess.run([lambdas], feed_dict={X:X_event})
    )
    H = H.reshape(H.shape[1],H.shape[2])
    for hs in H:
        T_pred_test.append(np.argmax(hs)+1)

    mae = np.mean(np.abs(T_event-T_pred_test))
    T_pred_test = np.array(T_pred_test)
    print "epoch %s"%n_epoch
    print "early detection: ", np.sum(T_pred_test<=T_event)
    print "delayed detection: ", np.sum(T_pred_test>T_event)
    print "early detection steps: ", np.mean(T_event[T_pred_test<=T_event]-T_pred_test[T_pred_test<=T_event])
    print
    print

    print "comparing with cut points: "
    print "after: ", np.sum(cut_points<T_pred_test)
    print "before: ", np.sum(cut_points > T_pred_test)
    print "= : ", np.sum(cut_points == T_pred_test)

    print "after cut_points steps: ", np.mean(T_pred_test[cut_points<T_pred_test]-cut_points[cut_points<T_pred_test])
    print
    print

    # print("epoch %s: %s %s %s" %(n_epoch, np.mean(rank_batch_loss), np.mean(batch_mae), mae))
    print cut_points[0:20]
    print T_pred_test[0:20]
    print T_event[0:20]
    print
    print

    acc_pair_test, usr2T_test = acc_pair_wtte(T_event)
    count = 0
    for p in acc_pair_test:
        if T_pred_test[p[0]] < T_pred_test[p[1]]:
            count += 1
    CI = count/float(len(acc_pair_test))
    # print("epoch %s: %s %s %s %s" %(n_epoch, np.mean(rank_batch_loss), np.mean(mle_batch_loss), mae, CI))
