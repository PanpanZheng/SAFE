import numpy as np
import sys
sys.path.append("../")
from safdKit import ran_seed



# X_ben = np.load("./v8/X_v8_5_20_Ben.npy")
# X_van = np.load("./v8/X_v8_5_20_Van.npy")

X_ben = np.load("./v9/X_v9_5_20_Ben.npy")
X_van = np.load("./v9/X_v9_5_20_Van.npy")


T_ben = list()
for usr in X_ben:
    T_ben.append(len(usr))

T_van = list()
for usr in X_van:
    T_van.append(len(usr))

T_ben = np.asarray(T_ben)
T_van = np.asarray(T_van)


C_ben = np.zeros(X_ben.shape[0])
C_van = np.ones(X_van.shape[0])


X = list()
T = list()
C = list()

for i in np.arange(12, 21):
    X_cen = X_ben[T_ben == i].tolist()
    X_cen = np.array(X_cen)
    # print X_cen.shape
    C_cen = C_ben[T_ben == i]
    T_cen = T_ben[T_ben == i]
    if X_cen.shape[0] > 100:
        s = ran_seed(X_cen.shape[0])
        X.extend((X_cen[s])[:100])
        C.extend((C_cen[s])[:100])
        T.extend((T_cen[s])[:100])
    else:
        X.extend(X_cen)
        C.extend(C_cen)
        T.extend(T_cen)

    X_uncen =  X_van[T_van == i].tolist()
    X_uncen = np.array(X_uncen)
    # print X_uncen.shape
    C_uncen = C_van[T_van == i]
    T_uncen = T_van[T_van == i]

    if X_uncen.shape[0] > 100:
        s = ran_seed(X_uncen.shape[0])
        X.extend((X_uncen[s])[:100])
        C.extend((C_uncen[s])[:100])
        T.extend((T_uncen[s])[:100])
    else:
        X.extend(X_uncen)
        C.extend(C_uncen)
        T.extend(T_uncen)

X, T, C = np.array(X), np.array(T), np.array(C)

train_num = int(0.7*X.shape[0])
valid_num = int(0.1*X.shape[0])
test_num = X.shape[0]-train_num-valid_num

s = ran_seed(X.shape[0])
X_train = X[s][:train_num]
T_train = T[s][:train_num]
C_train = C[s][:train_num]

X_valid = X[s][train_num:train_num+valid_num]
T_valid = T[s][train_num:train_num+valid_num]
C_valid = C[s][train_num:train_num+valid_num]

X_test = X[s][train_num+valid_num:]
T_test = T[s][train_num+valid_num:]
C_test = C[s][train_num+valid_num:]


np.save("X_train",X_train)
np.save("X_valid",X_valid)
np.save("X_test",X_test)

np.save("T_train",T_train)
np.save("T_valid",T_valid)
np.save("T_test",T_test)

np.save("C_train",C_train)
np.save("C_valid",C_valid)
np.save("C_test",C_test)

exit(0)