
import tensorflow as tf
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize
import sys
sys.path.append("../")
from safdKit import ran_seed, concordance_index, build_data, load_file, acc_pair_wtte




dest_path = "../Data/"
train = load_file(dest_path + 'train.csv')
test_x = load_file(dest_path + 'test_x.csv')
test_y = load_file(dest_path + 'test_y.csv')

#
all_x = np.concatenate((train[:, 2:26], test_x[:, 2:26]))
all_x = normalize(all_x, axis=0)

train[:, 2:26] = all_x[0:train.shape[0], :]
test_x[:, 2:26] = all_x[train.shape[0]:, :]

train[:, 0:2] -= 1
test_x[:, 0:2] -= 1

# exit(0)
max_time = 23
train_x, T, C = build_data(train[:, 0], train[:, 1], train[:, 2:26],max_time)

n_samples = len(train_x)
s = np.arange(n_samples)
np.random.shuffle(s)
train_x, T, C = train_x[s], T[s], C[s]



np.save(dest_path + "X_train_wtte.npy", train_x[0:1000])
np.save(dest_path + "T_train_wtte.npy", T[0:1000])
np.save(dest_path + "C_train_wtte.npy", C[0:1000])

np.save(dest_path + "X_test_wtte.npy", train_x[1000:1500])
np.save(dest_path + "T_test_wtte.npy", T[1000:1500])
np.save(dest_path + "C_test_wtte.npy", C[1000:1500])


