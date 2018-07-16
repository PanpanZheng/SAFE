import numpy as np
import json
from collections import defaultdict
from safdKit import sample_shuffle_uspv, order_grpah, acc_pair
from keras.preprocessing.sequence import pad_sequences


source_path = "../Twitter/Data/"
dest_path = "./ori_data/"

# users & labels
users, labels =  np.load(source_path + "extract_data/users.npy"), np.load(source_path + "extract_data/labels.npy")
uncen_usr, cen_usr = (users[labels == 0]).tolist(), (users[labels == 1]).tolist()

#1. Observed Time
followers_count = json.load(open(source_path + "extract_data/followers_count.json", "r"))
usr_2_T = defaultdict(list)

for usr, followers in followers_count.items():
    usr_2_T[usr] = len(followers)

#2. construct survival analysis data set with 'uncensored & censored' users above.

friends_count = json.load(open(source_path + "extract_data/friends_count.json", "r"))
listed_count = json.load(open(source_path + "extract_data/listed_count.json", "r"))
favourites_count = json.load(open(source_path + "extract_data/favourites_count.json", "r"))
statuses_count = json.load(open(source_path + "extract_data/statuses_count.json", "r"))

uncen_usr_seq, cen_usr_seq = defaultdict(list), defaultdict(list)

for usr in uncen_usr:
    try:
        uncen_usr_seq[usr]\
            = [[map(int, followers_count[usr]), map(int,friends_count[usr]),
                map(int, listed_count[usr]),map(int, favourites_count[usr]),
                map(int,statuses_count[usr])], usr_2_T[usr],1]
    except ValueError:
        uncen_usr.remove(usr)

for usr in cen_usr:
    try:
        cen_usr_seq[usr]\
            = [[map(int,followers_count[usr]), map(int,friends_count[usr]),
                map(int,listed_count[usr]),map(int,favourites_count[usr]),
                map(int,statuses_count[usr])], usr_2_T[usr],0]
    except ValueError:
        cen_usr.remove(usr)

for u, v in uncen_usr_seq.items():
    uncen_usr_seq[u][0] = pad_sequences(v[0], maxlen=22, padding="post")

for u, v in cen_usr_seq.items():
    cen_usr_seq[u][0] = pad_sequences(v[0], maxlen=22, padding="post")


UU_uncen = list()
UU_cen = list()

X_uncen = list()
X_cen = list()

T_uncen = list()
T_cen = list()

C_uncen = list()
C_cen = list()

for u, v in uncen_usr_seq.items():
    UU_uncen.append(u)
    X_uncen.append(np.array(v[0]).transpose())
    T_uncen.append(v[1])
    C_uncen.append(v[2])

for u, v in cen_usr_seq.items():
    UU_cen.append(u)
    X_cen.append(np.array(v[0]).transpose())
    T_cen.append(v[1])
    C_cen.append(v[2])

def ran_seed(N):
    s = np.arange(N)
    np.random.shuffle(s)
    return s

UU_uncen, X_uncen, T_uncen, C_uncen = np.array(UU_uncen), np.array(X_uncen), np.array(T_uncen), np.array(C_uncen)
s_uncen = ran_seed(UU_uncen.shape[0])
UU_uncen, X_uncen, T_uncen, C_uncen = UU_uncen[s_uncen], X_uncen[s_uncen], T_uncen[s_uncen], C_uncen[s_uncen]

UU_uncen_sampling = list()
X_uncen_sampling = list()
T_uncen_sampling = list()
C_uncen_sampling = list()



for t in set(T_uncen):
    UU_uncen_sampling.extend(UU_uncen[T_uncen == t])
    X_uncen_sampling.extend(X_uncen[T_uncen == t])
    T_uncen_sampling.extend(T_uncen[T_uncen == t])
    C_uncen_sampling.extend(C_uncen[T_uncen == t])

UU_uncen_sampling, X_uncen_sampling, \
                T_uncen_sampling, C_uncen_sampling = np.array(UU_uncen_sampling), np.array(X_uncen_sampling), \
                                                      np.array(T_uncen_sampling), np.array(C_uncen_sampling)


U_train, X_train, T_train, C_train = list(), list(), list(), list()
U_valid, X_valid, T_valid, C_valid = list(), list(), list(), list()
U_test, X_test, T_test, C_test = list(), list(), list(), list()

train_ratio = 0.7
valid_ratio = 0.1

for t in set(T_uncen_sampling):

    UU_temp = UU_uncen_sampling[T_uncen_sampling == t]
    U_train.extend(UU_temp[0:int(UU_temp.shape[0]*train_ratio)])
    U_valid.extend(UU_temp[int(UU_temp.shape[0]*train_ratio):int(UU_temp.shape[0]*(train_ratio+valid_ratio))])
    U_test.extend(UU_temp[int(UU_temp.shape[0]*(train_ratio+valid_ratio)):])

    X_temp = X_uncen_sampling[T_uncen_sampling == t]
    X_train.extend(X_temp[0:int(X_temp.shape[0]*train_ratio)])
    X_valid.extend(X_temp[int(X_temp.shape[0]*train_ratio):int(X_temp.shape[0]*(train_ratio+valid_ratio))])
    X_test.extend(X_temp[int(X_temp.shape[0]*(train_ratio+valid_ratio)):])

    T_temp = T_uncen_sampling[T_uncen_sampling == t]
    T_train.extend(T_temp[0:int(T_temp.shape[0]*train_ratio)])
    T_valid.extend(T_temp[int(T_temp.shape[0]*train_ratio):int(T_temp.shape[0]*(train_ratio+valid_ratio))])
    T_test.extend(T_temp[int(T_temp.shape[0]*(train_ratio+valid_ratio)):])

    C_temp = C_uncen_sampling[T_uncen_sampling == t]
    C_train.extend(C_temp[0:int(C_temp.shape[0]*train_ratio)])
    C_valid.extend(C_temp[int(C_temp.shape[0]*train_ratio):int(C_temp.shape[0]*(train_ratio+valid_ratio))])
    C_test.extend(C_temp[int(C_temp.shape[0]*(train_ratio+valid_ratio)):])


UU_cen, X_cen, T_cen, C_cen = np.array(UU_cen), np.array(X_cen), np.array(T_cen), np.array(C_cen)
s_cen = ran_seed(UU_cen.shape[0])
UU_cen, X_cen, T_cen, C_cen = UU_cen[s_cen], X_cen[s_cen], T_cen[s_cen], C_cen[s_cen]

s_num = UU_uncen_sampling.shape[0]
UU_cen, X_cen, T_cen, C_cen = UU_cen[:s_num], X_cen[:s_num], T_cen[:s_num], C_cen[:s_num]

U_train.extend(UU_cen[0:int(s_num*train_ratio)])
U_valid.extend(UU_cen[int(s_num*train_ratio):int(s_num*(train_ratio+valid_ratio))])
U_test.extend(UU_cen[int(s_num*(train_ratio+valid_ratio)):])

X_train.extend(X_cen[0:int(s_num*train_ratio)])
X_valid.extend(X_cen[int(s_num*train_ratio):int(s_num*(train_ratio+valid_ratio))])
X_test.extend(X_cen[int(s_num*(train_ratio+valid_ratio)):])

T_train.extend(T_cen[0:int(s_num*train_ratio)])
T_valid.extend(T_cen[int(s_num*train_ratio):int(s_num*(train_ratio+valid_ratio))])
T_test.extend(T_cen[int(s_num*(train_ratio+valid_ratio)):])

C_train.extend(C_cen[0:int(s_num*train_ratio)])
C_valid.extend(C_cen[int(s_num*train_ratio):int(s_num*(train_ratio+valid_ratio))])
C_test.extend(C_cen[int(s_num*(train_ratio+valid_ratio)):])

U_train, X_train, T_train, C_train = np.array(U_train), np.array(X_train), np.array(T_train), np.array(C_train)
U_valid, X_valid, T_valid, C_valid = np.array(U_valid), np.array(X_valid), np.array(T_valid), np.array(C_valid)
U_test, X_test, T_test, C_test = np.array(U_test), np.array(X_test), np.array(T_test), np.array(C_test)


# U_train, X_train, T_train, C_train = U_train[0:3000], X_train[0:3000], T_train[0:3000], C_train[0:3000]
# U_valid, X_valid, T_valid, C_valid = U_valid[0:1200], X_valid[0:1200], T_valid[0:1200], C_valid[0:1200]
# U_test, X_test, T_test, C_test = U_test[0:1800], X_test[0:1800],T_test[0:1800], C_test[0:1800]

# print U_train.shape, U_valid.shape, U_test.shape
# print X_train.shape, X_valid.shape, X_test.shape
# print T_train.shape, T_valid.shape, T_test.shape
# print C_train.shape, C_valid.shape, C_test.shape

np.save(dest_path + "U_train",U_train)
np.save(dest_path + "X_train",X_train)
np.save(dest_path + "T_train",T_train)
np.save(dest_path + "C_train",C_train)

np.save(dest_path + "U_valid",U_valid)
np.save(dest_path + "X_valid",X_valid)
np.save(dest_path + "T_valid",T_valid)
np.save(dest_path + "C_valid",C_valid)

np.save(dest_path + "U_test",U_test)
np.save(dest_path + "X_test",X_test)
np.save(dest_path + "T_test",T_test)
np.save(dest_path + "C_test",C_test)

exit(0)