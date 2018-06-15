import numpy as np
import json
from collections import defaultdict
from safdKit import sample_shuffle_uspv, order_grpah
from keras.preprocessing.sequence import pad_sequences


source_path = "../Twitter/Data/"

# users & labels
users, labels =  np.load(source_path + "extract_data/users.npy"), np.load(source_path + "extract_data/labels.npy")


#1. pick up 'uncensored' & 'censored' users.

followers_count = json.load(open(source_path + "extract_data/followers_count.json", "r"))

sus_seq = defaultdict(list)

max_len = 0
for usr, followers in followers_count.items():
    sus_seq[len(followers)].append(usr)
    if len(followers) > max_len:
        max_len = len(followers)

sampling_points = [3,4,11,13,18,20]

sus_usr_cdd = list()
for t in sampling_points:
    sus_usr_cdd += sus_seq[t]

uncen_usr = dict()
for usr in sample_shuffle_uspv(sus_usr_cdd,4000):
    for t in sus_seq.keys():
        if usr in set(sus_seq[t]):
            uncen_usr[usr] = t
            break

cen_usr_las_tim = dict()
for usr in sample_shuffle_uspv(sus_seq[max_len],3000):
    cen_usr_las_tim[usr] = max_len

cen_usr_mis_cdd = set(followers_count.keys())-set(uncen_usr.keys()+cen_usr_las_tim.keys())

cum_miss = list()
cen_usr_mis = dict()
for t in sorted(sus_seq.keys()):
    tmp_miss = sample_shuffle_uspv(list(cen_usr_mis_cdd-set(sus_seq[t])-set(cum_miss)), 500)
    cen_usr_mis_cdd -= set(sus_seq[t])
    cum_miss += tmp_miss

    if t in sampling_points:
        for usr in tmp_miss:
            cen_usr_mis[usr] = t

dest_path = "./Data/"
with open(dest_path + 'uncen_usr.json', 'w') as f:
    json.dump(uncen_usr, f)

with open(dest_path + 'cen_usr_las_tim.json', 'w') as f:
    json.dump(cen_usr_las_tim, f)

with open(dest_path + 'cen_usr_mis.json', 'w') as f:
    json.dump(cen_usr_mis, f)

with open(dest_path + 'sus_seq.json', 'w') as f:
    json.dump(sus_seq, f)



#2. construct survival analysis data set with 'uncensored & censored' users above.

source_path = "../Twitter/Data/"

# followers_count = json.load(open(source_path + "extract_data/followers_count.json", "r"))
friends_count = json.load(open(source_path + "extract_data/friends_count.json", "r"))
listed_count = json.load(open(source_path + "extract_data/listed_count.json", "r"))
favourites_count = json.load(open(source_path + "extract_data/favourites_count.json", "r"))
statuses_count = json.load(open(source_path + "extract_data/statuses_count.json", "r"))


uncen_usr_seq = defaultdict(list)
cen_usr_las_tim_seq = defaultdict(list)
cen_usr_mis_seq = defaultdict(list)


for usr in uncen_usr.keys():
    uncen_usr_seq[usr]\
        = [[map(int, followers_count[usr]), map(int,friends_count[usr]),
            map(int, listed_count[usr]),map(int, favourites_count[usr]),map(int,statuses_count[usr])],
           uncen_usr[usr],
           1]

for usr in cen_usr_las_tim.keys():
    cen_usr_las_tim_seq[usr]\
        = [[map(int,followers_count[usr]), map(int,friends_count[usr]), map(int,listed_count[usr]),
            map(int,favourites_count[usr]),map(int,statuses_count[usr])],
           cen_usr_las_tim[usr],
           0]

for usr in cen_usr_mis.keys():
    cen_usr_mis_seq[usr]\
        = [[map(int, followers_count[usr][:cen_usr_mis[usr]]), map(int,friends_count[usr][:cen_usr_mis[usr]]), map(int,listed_count[usr][:cen_usr_mis[usr]]),
            map(int, favourites_count[usr][:cen_usr_mis[usr]]),map(int, statuses_count[usr][:cen_usr_mis[usr]])],
           cen_usr_mis[usr],
           0]

with open(dest_path + 'uncen_usr_seq.json', 'w') as f:
    json.dump(uncen_usr_seq, f)

with open(dest_path + 'cen_usr_las_tim_seq.json', 'w') as f:
    json.dump(cen_usr_las_tim_seq, f)

with open(dest_path + 'cen_usr_mis_seq.json', 'w') as f:
    json.dump(cen_usr_mis_seq, f)


# uncen_usr_seq = json.load(open("../Data/uncen_usr_seq.json", "r"))
# cen_usr_las_tim_seq = json.load(open("../Data/cen_usr_las_tim_seq.json", "r"))
# cen_usr_mis_seq = json.load(open("../Data/cen_usr_mis_seq.json", "r"))


for u, v in uncen_usr_seq.items():
    uncen_usr_seq[u][0] = pad_sequences(v[0], maxlen=22, padding="post")

for u, v in cen_usr_las_tim_seq.items():
    cen_usr_las_tim_seq[u][0] = pad_sequences(v[0], maxlen=22, padding="post")

for u, v in cen_usr_mis_seq.items():
    cen_usr_mis_seq[u][0] = pad_sequences(v[0], maxlen=22, padding="post")


# with open(dest_path + 'uncen_usr_seq_pad.json', 'w') as f:
#     json.dump(uncen_usr_seq, f)
#
# with open(dest_path + 'cen_usr_las_tim_seq_pad.json', 'w') as f:
#     json.dump(cen_usr_las_tim_seq, f)
#
# with open(dest_path + 'cen_usr_mis_seq_pad.json', 'w') as f:
#     json.dump(cen_usr_mis_seq, f)

# np.save(dest_path + "uncen_usr_seq_pad", uncen_usr_seq.values())
# np.save(dest_path + "cen_usr_las_tim_seq_pad", cen_usr_las_tim_seq.values())
# np.save(dest_path + "cen_usr_mis_seq_pad", cen_usr_mis_seq.values())

# uncen_usr_seq_pad = np.load(dest_path + "uncen_usr_seq_pad.npy")
# cen_usr_las_tim_seq_pad = np.load(dest_path + "cen_usr_las_tim_seq_pad.npy")
# cen_usr_mis_seq_pad = np.load(dest_path + "cen_usr_mis_seq_pad.npy")

UU = list()

X_event = list()
X_censor = list()

T_event = list()
T_censor = list()

C_event = list()
C_censor = list()

for u, v in uncen_usr_seq.items():
    X_event.append(np.array(v[0]).transpose())
    T_event.append(v[1])
    C_event.append(v[2])
    UU.append(u)

for u, v in cen_usr_las_tim_seq.items():
    X_censor.append(np.array(v[0]).transpose())
    T_censor.append(v[1])
    C_censor.append(v[2])
    UU.append(u)

for u, v in cen_usr_mis_seq.items():
    X_censor.append(np.array(v[0]).transpose())
    T_censor.append(v[1])
    C_censor.append(v[2])
    UU.append(u)

# for usr in uncen_usr_seq_pad:
#     X_event.append(np.array(usr[0]).transpose())
#     T_event.append(usr[1])
#     C_event.append(usr[2])
#
# for usr in cen_usr_las_tim_seq_pad:
#     X_censor.append(np.array(usr[0]).transpose())
#     T_censor.append(usr[1])
#     C_censor.append(usr[2])
#
# for usr in cen_usr_mis_seq_pad:
#     X_censor.append(np.array(usr[0]).transpose())
#     T_censor.append(usr[1])
#     C_censor.append(usr[2])

X = X_event + X_censor
T = T_event + T_censor
C = C_event + C_censor

# X ,T, C = np.array(X), np.array(T), np.array(C)

X_train = X[0:3000] + X[4000:5000] + X[7000:8000]
T_train = T[0:3000] + T[4000:5000] + T[7000:8000]
C_train = C[0:3000] + C[4000:5000] + C[7000:8000]
U_train = UU[0:3000] + UU[4000:5000] + UU[7000:8000]

X_test = X[3000:4000] + X[5000:5500] + X[8000:8500]
T_test = T[3000:4000] + T[5000:5500] + T[8000:8500]
C_test = C[3000:4000] + C[5000:5500] + C[8000:8500]
U_test = UU[3000:4000] + UU[5000:5500] + UU[8000:8500]

np.save(dest_path + "X_train",X_train)
np.save(dest_path + "T_train",T_train)
np.save(dest_path + "C_train",C_train)
np.save(dest_path + "U_train",U_train)

np.save(dest_path + "X_test",X_test)
np.save(dest_path + "T_test",T_test)
np.save(dest_path + "C_test",C_test)
np.save(dest_path + "U_test",U_test)

uncen_usr_f = json.load(open("./Data/uncen_usr.json", "r"))
cen_usr_las_tim_f = json.load(open("./Data/cen_usr_las_tim.json", "r"))
cen_usr_mis_f = json.load(open("./Data/cen_usr_mis.json", "r"))

ord_gra = order_grpah(uncen_usr_f, cen_usr_las_tim_f, cen_usr_mis_f)

usr_index = dict()

for i, u in enumerate(U_train):
    usr_index[u] = i

with open(dest_path + 'usr_id_dict_train.json', 'w') as f:
    json.dump(usr_index, f)

acc_pair = defaultdict(list)
for u in U_train[0:3000]:
    if u in ord_gra:
        for u_desc in ord_gra[u]:
            if u_desc in usr_index:
                acc_pair[usr_index[u]].append(usr_index[u_desc])

with open(dest_path + 'acc_pair.json', 'w') as f:
    json.dump(acc_pair, f)

usr_index_test = dict()
for i, u in enumerate(U_test):
    usr_index_test[u] = i
with open(dest_path + 'usr_id_dict_test.json', 'w') as f:
    json.dump(usr_index_test, f)


exit(0)