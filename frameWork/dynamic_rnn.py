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
source1 = "../Data3/"
source2 = "../Data4/"

X_train_12 = np.load(source1 + "X_train_12.npy")
X_test_12 = np.load(source1 + "X_test_12.npy")

X_train_17 = np.load(source1 + "X_train_17.npy")
X_test_17 = np.load(source1 + "X_test_17.npy")

X_train_18 = np.load(source1 + "X_train_18.npy")
X_test_18 = np.load(source1 + "X_test_18.npy")

X_train_19 = np.load(source1 + "X_train_19.npy")
X_test_19 = np.load(source1 + "X_test_19.npy")

X_train_1_21 = np.load(source1 + "X_train_21.npy")
X_test_1_21 = np.load(source1 + "X_test_21.npy")

X_train_2_21 = np.load(source2 + "X_train_21.npy")
X_train_9 = np.load(source2 + "X_train_9.npy")
X_train_8 = np.load(source2 + "X_train_8.npy")
X_train_7 = np.load(source2 + "X_train_7.npy")
X_train_6 = np.load(source2 + "X_train_6.npy")
X_train_5 = np.load(source2 + "X_train_5.npy")
X_train_4 = np.load(source2 + "X_train_4.npy")
X_train_3 = np.load(source2 + "X_train_3.npy")
X_train_2 = np.load(source2 + "X_train_2.npy")
X_train_1 = np.load(source2 + "X_train_1.npy")

X_test_2_21 = np.load(source2 + "X_test_21.npy")
X_test_9 =np.load(source2 + "X_test_9.npy")
X_test_8 =np.load(source2 + "X_test_8.npy")
X_test_7 =np.load(source2 + "X_test_7.npy")
X_test_6 =np.load(source2 + "X_test_6.npy")
X_test_5 =np.load(source2 + "X_test_5.npy")
X_test_4 =np.load(source2 + "X_test_4.npy")
X_test_3 =np.load(source2 + "X_test_3.npy")
X_test_2 =np.load(source2 + "X_test_2.npy")
X_test_1 =np.load(source2 + "X_test_1.npy")



X_train_21 = np.concatenate((X_train_1_21,X_train_2_21))
X_test_21 = np.concatenate((X_test_1_21,X_test_2_21))

# print X_test_1.shape, X_test_2.shape, X_test_3.shape, X_test_4.shape, X_test_5.shape,X_test_6.shape, \
#     X_test_7.shape,X_test_8.shape, X_test_9.shape, X_test_12.shape, X_test_17.shape, X_test_18.shape,X_test_19.shape,X_test_21.shape
#
# exit(0)


session = tf.Session()
session.run(tf.global_variables_initializer())

num_epoches = 700

for n_epoch in range(num_epoches):

    for i in [1,2,3,4,5,6,7,8,9,12,17,18,19,21]:

        if i == 1:
            bat_X = X_train_1
        elif i == 2:
            bat_X = X_train_2
        elif i == 3:
            bat_X = X_train_3
        elif i == 4:
            bat_X = X_train_4
        elif i == 5:
            bat_X = X_train_5
        elif i == 6:
            bat_X = X_train_6
        elif i == 7:
            bat_X = X_train_7
        elif i == 8:
            bat_X = X_train_8
        elif i == 9:
            bat_X = X_train_9
        elif i == 12:
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

        _, _mse_loss = session.run([train_supervised_op, mse_loss],feed_dict={
                                X: bat_X,
                                y_true: bat_y,
                                batch_size: bat_size
        })

    print "epoch: ", n_epoch, _mse_loss


# exit(0)
_y_pred_test = list()

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 18, 19,21]:

    if i == 1:
        bat_X = X_test_1
    elif i == 2:
        bat_X = X_test_2
    elif i == 3:
        bat_X = X_test_3
    elif i == 4:
        bat_X = X_test_4
    elif i == 5:
        bat_X = X_test_5
    elif i == 6:
        bat_X = X_test_6
    elif i == 7:
        bat_X = X_test_7
    elif i == 8:
        bat_X = X_test_8
    elif i == 9:
        bat_X = X_test_9
    elif i == 12:
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

_y_test = np.concatenate((np.ones(632),np.zeros(632)))
_y_pred_test = np.array(_y_pred_test)
_y_pred_test = (np.array(_y_pred_test)>0.5).astype(int)
acc = accuracy_score(_y_test, _y_pred_test)
print acc

exit(0)