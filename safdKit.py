import numpy as np
import json
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from sklearn import tree, ensemble, neighbors, svm, covariance
import random
np.seterr(divide='ignore', invalid='ignore')


def sample_shuffle_uspv(X,n):

    n_samples = len(X)
    s = np.arange(n_samples)
    np.random.shuffle(s)
    return [X[i] for i in s][0:n]


def load_twitter():

    uncen_usr_data = json.load(open("./Data/uncen_usr_data.json", "r"))
    cen_usr_las_tim_data = json.load(open("./Data/cen_usr_las_tim_data.json", "r"))
    cen_usr_mis_data = json.load(open("./Data/cen_usr_mis_data.json", "r"))

    uncen_usr_data_keys = list()
    uncen_usr_data_values = list()

    for key, value in uncen_usr_data.items():
        uncen_usr_data_keys.append(key)
        uncen_usr_data_values.append(value)

    cen_usr_las_tim_data_keys = list()
    cen_usr_las_tim_data_values = list()

    for key, value in cen_usr_las_tim_data.items():
        cen_usr_las_tim_data_keys.append(key)
        cen_usr_las_tim_data_values.append(value)

    cen_usr_mis_data_keys = list()
    cen_usr_mis_data_values = list()

    for key, value in cen_usr_mis_data.items():
        cen_usr_mis_data_keys.append(key)
        cen_usr_mis_data_values.append(value)

    uncen_usr_df = df(uncen_usr_data_values,
                             columns=['follower_#', 'friend_#', 'member_#', 'favor_#', 'tweet_#', 'Survival', 'Event'])
    cen_usr_las_tim_df = df(cen_usr_las_tim_data_values,
                                   columns=['follower_#', 'friend_#', 'member_#', 'favor_#', 'tweet_#', 'Survival',
                                            'Event'])
    cen_usr_mis_df = df(cen_usr_mis_data_values,
                               columns=['follower_#', 'friend_#', 'member_#', 'favor_#', 'tweet_#', 'Survival',
                                        'Event'])
    return pd.concat([uncen_usr_df.iloc[0:3000], cen_usr_las_tim_df.iloc[0:2000], cen_usr_mis_df.iloc[0:2000]]).convert_objects(convert_numeric=True), \
           pd.concat([uncen_usr_df.iloc[3000:4000], cen_usr_las_tim_df.iloc[2000:3000], cen_usr_mis_df.iloc[2000:3000]]).convert_objects(convert_numeric=True), \
           uncen_usr_data_keys[3000:4000] + cen_usr_las_tim_data_keys[2000:3000] + cen_usr_mis_data_keys[2000:3000]

def order_grpah(uncen_usr, cen_usr_las_tim, cen_usr_mis):

    order_graph = defaultdict(list)
    for usr_event in uncen_usr.keys():

        for usr_po_desc in uncen_usr.keys():
            if uncen_usr[usr_event] < uncen_usr[usr_po_desc]:
                order_graph[usr_event].append(usr_po_desc)

        for usr_po_desc in cen_usr_las_tim.keys():
            if uncen_usr[usr_event] < cen_usr_las_tim[usr_po_desc]:
                order_graph[usr_event].append(usr_po_desc)

        for usr_po_desc in cen_usr_mis.keys():
            if uncen_usr[usr_event] < cen_usr_mis[usr_po_desc]:
                order_graph[usr_event].append(usr_po_desc)
    return order_graph


def acc_pair_wtte(eve_T):

    usr2T = dict()
    for i, T in enumerate(eve_T):
        usr2T[i] = T

    acc_pair = list()
    for usr_p in usr2T.keys():
        for usr_d in usr2T.keys():
            if usr_p != usr_d:
                if usr2T[usr_p] < usr2T[usr_d]:
                    acc_pair.append([usr_p,usr_d])
    return np.array(acc_pair), usr2T


def acc_pair(T, C):

    usr2T = dict()
    for i, t in enumerate(T):
        usr2T[i] = t

    i_uncen = np.arange(T.shape[0])[C==1]

    acc_pair = list()
    for usr_p in i_uncen:
        for usr_d in np.arange(T.shape[0]):
            if usr_p != usr_d:
                if usr2T[usr_p] < usr2T[usr_d]:
                    acc_pair.append([usr_p,usr_d])
    return np.array(acc_pair), usr2T



def get_survival_time(sur_pro, thrld, time_stamp, usr_ts):

    n_row, n_col = sur_pro.shape
    eva_sur_tim = dict()
    for i in range(n_row):
        flag = True
        for j in range(n_col):
            if sur_pro[i,j] < thrld:
                eva_sur_tim[usr_ts[i]] = time_stamp[j-1]
                flag = False
                break
        if flag:
            eva_sur_tim[usr_ts[i]] = time_stamp[-1]
    return eva_sur_tim


def concordance_index(ord_gra, f_sur_tim, cons_uncen, cons_t):

    edge_n = 0
    corr_pred_count = 0
    test_uncen_usr = list(set(ord_gra.keys()).intersection(f_sur_tim.keys()))
    for par in test_uncen_usr:
        for dec in list(set(ord_gra[par]).intersection(f_sur_tim.keys())):
            edge_n += 1
            if cons_uncen:
                par_pivot = cons_t
            else:
                par_pivot = f_sur_tim[par]
            if par_pivot < f_sur_tim[dec]:
                corr_pred_count += 1

    return corr_pred_count/float(edge_n), corr_pred_count, edge_n


def mean_squared_error(uncen_usr, f_sur_tim, cons_uncen, cons_t):

    err_coll = list()
    test_uncen_usr = list(set(uncen_usr.keys()).intersection(set(f_sur_tim.keys())))
    for usr in test_uncen_usr:
        if cons_uncen:
            eva_t = cons_t
        else:
            eva_t = f_sur_tim[usr]
        err_coll.append(eva_t-uncen_usr[usr])
    return np.mean(err_coll), np.std(err_coll)


def hist_statistics(uncen_usr, cen_usr_las_tim, cen_usr_mis):

    hist_uncen_usr = dict()
    hist_cen_usr_las_tim = dict()
    hist_cen_usr_mis = dict()
    hist_total = dict()

    for i in range(22):
        hist_total[i + 1] = 0
        hist_uncen_usr[i+1] = 0
        hist_cen_usr_las_tim[i+1] = 0
        hist_cen_usr_mis[i+1] = 0

    for t in uncen_usr.values():
        hist_uncen_usr[int(t)] += 1
        hist_total[int(t)] += 1

    for t in cen_usr_las_tim.values():
        hist_cen_usr_las_tim[int(t)] += 1
        hist_total[int(t)] += 1

    for t in cen_usr_mis.values():
        hist_cen_usr_mis[int(t)] += 1
        hist_total[int(t)] += 1

    return hist_total, hist_uncen_usr, hist_cen_usr_las_tim, hist_cen_usr_mis


def ErrHist(*args):

    N = 22
    U = args[0].values()
    C1 = args[1].values()
    C2 = args[2].values()

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, C1, width)
    p2 = plt.bar(ind, C2, width,
                 bottom=C1)
    p3 = plt.bar(ind, U, width,
                 bottom=C2)

    plt.ylabel('sample_#')
    plt.xlabel('time stamp')
    plt.title('censoring distribution')
    plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5',
                     'T6', 'T7', 'T8', 'T9', 'T10',
                     'T11', 'T12', 'T13', 'T14', 'T15',
                     'T16', 'T17', 'T18', 'T19', 'T20',
                     'T21', 'T22'))
    plt.yticks(np.arange(0, 3001, 100))
    plt.legend((p1[0], p2[0], p3[0]), ('cen_las_tim', 'cen_mis', 'uncensored'))

    plt.show()

def censor_distri(x, y1, y2, col_line1, col_line2, lab_word1, lab_word2):

    p1, = plt.plot(x, y1, col_line1, label=lab_word1)
    p2, = plt.plot(x, y2, col_line2, label=lab_word2)
    plt.legend(handles=[p1, p2])
    plt.xlabel('time stamp')
    plt.ylabel('percent')
    plt.show()


def ran_seed(N):
    s = np.arange(N)
    np.random.shuffle(s)
    return s



def build_data(engine,time,x,max_time):

    coll = []
    for i in range(100):
        max_engine_time = int(np.max(time[engine == i])) + 1
        ran_ind = np.arange(max_engine_time)
        np.random.shuffle(ran_ind)
        sca_len = min(int(np.floor(max_engine_time/10)),22)
        coll.append(
            (x[engine == i])[sorted(ran_ind[0:sca_len])]
        )

    coll_pad = list()
    T = list()
    C = list()
    for usr_rec in coll:
        rec_len = len(usr_rec)
        # print rec_len
        for j in range(rec_len):
            rec_step = (usr_rec[0:j+1]).tolist()
            for steps in range(len(rec_step), max_time):
                rec_step.append(np.zeros(24).tolist())
            # print rec_step
            coll_pad.append(rec_step)
            T.append(rec_len)
            C.append(1)

    return np.array(coll_pad), np.array(T), np.array(C)


def load_file(name):
    with open(name, 'r') as file:
        return np.loadtxt(file, delimiter=',')

def in_interval(itv, x):
    if itv[0]<=x<itv[1]:
        return True
    else:
        return False

def pick_up_pair(val_ran, acc_pair):
    pair_coll = list()
    for p in acc_pair:
        if in_interval(val_ran, p[0]) and in_interval(val_ran, p[1]):
            pair_coll.append(p)
    return pair_coll

def cut_seq(X,T,dim,ratio):
    for x, t in zip(X,T):
        for i in np.arange(int(t*ratio),t):
            x[i]= np.zeros(dim).tolist()
    # return X


def cut_seq_0(X,T,dim,ratio):
    X_cut = list()
    coll_len = list()
    cut_points = list()
    for x, t in zip(X, T):
        L = len(x)
        x = x[:int(t*ratio)]
        coll_len.append(t-int(t*ratio))
        cut_points.append(int(t*ratio))
        # last_ele = x[-1]
        for _ in np.arange(int(t*ratio),L):
            x.append(np.zeros(dim).tolist())
        X_cut.append(x)
    return np.array(X_cut), np.mean(coll_len), np.array(cut_points)


def cut_seq_last(X,T,ratio):
    X_cut = list()
    coll_len = list()
    for x, t in zip(X, T):
        L = len(x)
        x = x[:int(t*ratio)]
        last_ele = x[-1]
        coll_len.append(t - int(t * ratio))
        for _ in np.arange(int(t*ratio),L):
            x.append(last_ele)
        X_cut.append(x)
    return np.array(X_cut),np.mean(coll_len)


def cut_seq_mean(X,T,ratio):
    X_cut = list()
    coll_len = list()
    for x, t in zip(X, T):
        L = len(x)
        x = x[:int(t*ratio)]
        coll_len.append(t - int(t * ratio))
        mean_vac = np.mean(x,axis=0).tolist()
        for _ in np.arange(int(t*ratio),L):
            x.append(mean_vac)
        X_cut.append(x)
    return np.array(X_cut), np.mean(coll_len)


def pad_sequences_last_elem(X,L):
    X_p = list()
    for eles in X:
        for _ in np.arange(len(eles),L):
            eles.append(eles[-1])
        X_p.append(eles)
    return np.array(X_p)


def pad_sequences_mean(X,L):
    X_p = list()
    for eles in X:
        ave = np.mean(eles)
        for _ in np.arange(len(eles),L):
            eles.append(ave)
        X_p.append(eles)
    return np.array(X_p)

def survival(hs):
    return np.exp(-np.sum(hs))

def lambda2Survival(H):
    sur_coll = list()
    for i, h in enumerate(H):
        sur_coll.append(survival(H[0:i+1]))
    return np.array(sur_coll)


def k_NN(X,y):
	clf = neighbors.KNeighborsClassifier(n_neighbors=3)
	return clf.fit(X,y)

def decision_tree(X,y):
	clf = tree.DecisionTreeClassifier()
	return clf.fit(X, y)

def random_forest(X,y):
	clf = ensemble.RandomForestClassifier(n_estimators=10)
	return clf.fit(X,y)

def svm_svc(X,y):
	clf = svm.SVC()
	return clf.fit(X,y)

def svm_nusvc(X,y):
	clf = svm.NuSVC()
	return clf.fit(X,y)

def svm_linearsvc(X,y):
	clf = svm.LinearSVC()
	return clf.fit(X,y)

def svm_oneclass(X):
	clf = svm.OneClassSVM()
	return clf.fit(X)

def elliptic_envelope(X):
	clf = covariance.EllipticEnvelope()
	return clf.fit(X)

def upp_tri_mat(M):
    # n = M.shape[0]
    for i, r_m in enumerate(M):
        for j, m in enumerate(r_m):
            if j <= i:
                r_m[j] = 1
            else:
                break
    return np.transpose(M)

def draw_x_y(x,y):
    p1, = plt.plot(x, y, 'bo-', label='uncensor')
    plt.xlabel('time stamp')
    plt.ylabel('percent')
    plt.show()

def last_element(X, T):
    X_last = []
    for i, x_seq in enumerate(X):
        tmp = list()
        for j, x in enumerate(x_seq):
            if j < T[i]:
                tmp.append(x)
            else:
                tmp.append(np.mean(x_seq[:T[i]],axis=0).tolist())
        X_last.append(tmp)
    return np.array(X_last)

def remove_padding(X, T):
    X_last = list()
    for i, x_seq in enumerate(X):
        tmp = []
        for j, x in enumerate(x_seq):
            if j < T[i]:
                tmp.append(x.tolist())
            else:
                break
        X_last.append(tmp)
    return np.array(X_last)

def remove_padding_diff(X, T):
    X_last = list()
    for i, x_seq in enumerate(X):
        tmp = []
        for j, x in enumerate(x_seq):
            if j == 0:
                continue
            elif 0 < j < T[i]:
                tmp.append((x-x_seq[j-1]).tolist())
            else:
                break
        X_last.append(tmp)
    return np.array(X_last)




def minibatch(X, T, C, batch_size=16, n_input=5):

    minibatch_list_x = []
    minibatch_list_y = []
    minibatch_list_t = []
    minibatch_list_c = []
    # count = 0
    for t in np.unique(T):
        sub_X = X[np.where(T==t)]
        sub_C = C[np.where(T==t)]
        sub_T = T[np.where(T==t)]
        n_sub_batch = int(sub_X.shape[0]/batch_size)
        # count += n_sub_batch*batch_size
        for n in range(n_sub_batch):
            minibatch_list_x.append(np.asarray(list(sub_X[n*batch_size: (n+1)*batch_size])).reshape(batch_size,int(t),n_input))
            minibatch_list_t.append(np.asarray(list(sub_T[n*batch_size: (n+1)*batch_size])))
            minibatch_list_c.append(np.asarray(list(sub_C[n*batch_size: (n+1)*batch_size])))
            minibatch_list_y.append(
                np.logical_not(
                    np.asarray(list(sub_C[n*batch_size:(n+1)*batch_size]))
                ).astype(float)
            )
    assert(len(minibatch_list_x)==len(minibatch_list_y))
    c = list(zip(minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c))
    random.shuffle(c)
    minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c = zip(*c)
    return minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c

def minibatch_twitter(X, T, C, batch_size=16, n_input=5):

    minibatch_list_x = []
    minibatch_list_y = []
    minibatch_list_t = []
    minibatch_list_c = []
    # count = 0
    for t in np.unique(T):
        sub_X = X[np.where(T==t)]
        sub_C = C[np.where(T==t)]
        sub_T = T[np.where(T==t)]
        n_sub_batch = int(sub_X.shape[0]/batch_size)
        # count += n_sub_batch*batch_size
        for n in range(n_sub_batch):
            minibatch_list_x.append(np.asarray(list(sub_X[n*batch_size: (n+1)*batch_size])).reshape(batch_size,int(t),n_input))
            minibatch_list_t.append(np.asarray(list(sub_T[n*batch_size: (n+1)*batch_size])))
            minibatch_list_c.append(np.asarray(list(sub_C[n*batch_size: (n+1)*batch_size])))
            if t < 21:
                # minibatch_list_y.append(np.ones((batch_size,1)))
                minibatch_list_y.append(np.zeros(batch_size))
            else:
                # minibatch_list_y.append(np.zeros((batch_size, 1)))
                minibatch_list_y.append(np.ones(batch_size))
    assert(len(minibatch_list_x)==len(minibatch_list_y))
    c = list(zip(minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c))
    random.shuffle(c)
    minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c = zip(*c)
    return minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c



def minibatch_wiki(X, T, C, batch_size=16, n_input=5):

    minibatch_list_x = []
    minibatch_list_y = []
    minibatch_list_t = []
    minibatch_list_c = []
    # count = 0
    for t in np.unique(T):
        sub_X = X[np.where(T==t)]
        sub_C = C[np.where(T==t)]
        sub_T = T[np.where(T==t)]
        n_sub_batch = int(sub_X.shape[0]/batch_size)
        # count += n_sub_batch*batch_size
        for n in range(n_sub_batch):
            minibatch_list_x.append(np.asarray(list(sub_X[n*batch_size: (n+1)*batch_size])).reshape(batch_size,int(t),n_input))
            minibatch_list_t.append(np.asarray(list(sub_T[n*batch_size: (n+1)*batch_size])))
            minibatch_list_c.append(np.asarray(list(sub_C[n*batch_size: (n+1)*batch_size])))
            minibatch_list_y.append(
                np.logical_not(
                    np.asarray(list(sub_C[n*batch_size:(n+1)*batch_size]))
                ).astype(float)
            )
            # if t < 21:
            #     # minibatch_list_y.append(np.ones((batch_size,1)))
            #     minibatch_list_y.append(np.zeros(batch_size))
            # else:
            #     # minibatch_list_y.append(np.zeros((batch_size, 1)))
            #     minibatch_list_y.append(np.ones(batch_size))
    assert(len(minibatch_list_x)==len(minibatch_list_y))
    c = list(zip(minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c))
    random.shuffle(c)
    minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c = zip(*c)
    return minibatch_list_x, minibatch_list_y, minibatch_list_t, minibatch_list_c




def prec_reca_F1(L, I):

    _L = np.logical_not(L).astype(float)
    _I = np.logical_not(I).astype(float)

    TP = np.logical_and(L, I).astype(float)
    TN = np.logical_and(_L, _I).astype(float)
    FP = np.logical_and(_L, I).astype(float)
    FN = np.logical_and(L, _I).astype(float)

    precision = np.divide(np.sum(TP,axis=0),
                          np.add(np.sum(TP,axis=0),
                                 np.sum(FP,axis=0)))
    precision[np.where(np.isnan(precision))] = 0

    recall = np.divide(np.sum(TP,axis=0),
                          np.add(np.sum(TP,axis=0),
                                 np.sum(FN,axis=0)))
    recall[np.where(np.isnan(recall))] = 0

    F1 = np.multiply(2,
                np.divide(
                    np.multiply(precision, recall),
                    np.add(precision, recall)))
    F1[np.where(np.isnan(F1))] = 0

    return precision, recall, F1


def get_first_beat(x, y):

    x_1 = np.where(x==y)[0]
    x_2 = np.where(x==y)[1]

    d = defaultdict(list)
    for _x_1, _x_2 in zip(x_1, x_2):
        d[_x_1].append(_x_2)

    first_beat = list()
    for k, v in d.items():
        first_beat.append(v[0])
    first_beat = np.array(first_beat) + 1
    return first_beat


def twitter_dist(*args):
    N = 9
#     N = 10
    #     U = args[0].values()
    #     C1 = args[1].values()
    #     C2 = args[2].values()
    U = args[0]
    C1 = args[1]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, C1, width, color='y')
    #     p2 = plt.bar(ind, C2, width,
    #                  bottom=C1)
    p3 = plt.bar(ind, U, width,
                 bottom=C1, color='r')

    plt.ylabel('User number')
    plt.xlabel('Last observed timestamp')
    #     plt.title('Suspended-censor distribution for wiki')
#     plt.xticks(ind, ('T12', 'T13', 'T14', 'T15',
#                      'T16', 'T17', 'T18', 'T19', 'T20',
#                      'T21'))

    plt.xticks(ind, ('T12', 'T13', 'T14', 'T15',
                     'T16', 'T17', 'T18', 'T19', 'T20'))
    plt.yticks(np.arange(0, 500, 50))
    #     plt.yticks(np.arange(0, 3500, 500))
    plt.legend((p1[0], p3[0]), ('right-censored', 'Event'), prop={'size': 12})

    plt.show()


def wiki_dist(*args):
    N = 9
    #     U = args[0].values()
    #     C1 = args[1].values()
    #     C2 = args[2].values()
    U = args[0]
    C1 = args[1]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, C1, width)
    #     p2 = plt.bar(ind, C2, width,
    #                  bottom=C1)
    p3 = plt.bar(ind, U, width,
                 bottom=C1)

    plt.ylabel('Sample number')
    plt.xlabel('Editting sequence length')
    plt.title('Event-censor distribution for wiki')
    plt.xticks(ind, ('12', '13', '14', '15',
                     '16', '17', '18', '19', '20'))
    plt.yticks(np.arange(0, 500, 50))
    plt.legend((p1[0], p3[0]), ('right-censored', 'event'))

    plt.show()


def early_det(x, y):
    first_beat = []
    for _x, _y in zip(x, y):
        ear_det = False
        if np.where(_x==_y)[0].any():
            beat = True
            for i in range(np.where(_x==_y)[0][0],len(_x)):
                if _x[i] != _y[i]:
                    beat = False
                    break
            if beat:
                ear_det = True
        if ear_det:
            first_beat.append(np.where(_x==_y)[0][0] + 1)
    return np.asarray(first_beat)


def me_evaluation(N, men_means, women_means):

    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
#     rects1 = ax.bar(ind, child_means, width, color='r', yerr=child_std)
#     rects2 = ax.bar(ind, men_means, width, color='y', yerr=men_std)
    rects2 = ax.bar(ind, men_means, width, color='y')

    # women_means = (25, 32, 34, 20, 25)
    # women_std = (3, 5, 2, 3, 3)
#     rects3 = ax.bar(ind + width, women_means, width, color='b', yerr=women_std)
    rects3 = ax.bar(ind + width, women_means, width, color='b')

    # add some text for labels, title and axes ticks
#     ax.set_ylabel('Percentage of early-detected fraudsters')
    ax.set_ylabel('Early-detected timestamps')
    ax.set_xlabel('Suspended timestamp')
    #     ax.set_ylabel('Early Detected Instance Number')
    #     ax.set_title('Early Time Stamps by Groups')
    ax.set_xticks(ind + width/2)
#     ax.set_xticks(ind + width / 3)
    ax.set_xticklabels(('T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20'))

    ax.legend((rects2[0], rects3[0]), ('SAFE', 'M-LSTM'))

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    #             ax.text(rect.get_x() + rect.get_width()/3., 1.05*height,
                    '%.1f'%height,
                    ha='center', va='bottom')

#     autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.show()
