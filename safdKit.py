import numpy as np
import json
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences


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
    for x, t in zip(X, T):
        L = len(x)
        x = x[:int(t*ratio)]
        # last_ele = x[-1]
        for _ in np.arange(int(t*ratio),L):
            x.append(np.zeros(dim).tolist())
        X_cut.append(x)
    return np.array(X_cut)


def cut_seq_last(X,T,ratio):
    X_cut = list()
    for x, t in zip(X, T):
        L = len(x)
        x = x[:int(t*ratio)]
        last_ele = x[-1]
        for _ in np.arange(int(t*ratio),L):
            x.append(last_ele)
        X_cut.append(x)
    return np.array(X_cut)


def cut_seq_mean(X,T,ratio):
    X_cut = list()
    for x, t in zip(X, T):
        L = len(x)
        x = x[:int(t*ratio)]
        mean_vac = np.mean(x,axis=0).tolist()
        for _ in np.arange(int(t*ratio),L):
            x.append(mean_vac)
        X_cut.append(x)
    return np.array(X_cut)


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