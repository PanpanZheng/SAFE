import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pandas import DataFrame as df
import sys
import os
sys.path.append("../")
from sklearn import svm
import operator

from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from safdKit import load_twitter, prec_reca_F1, get_survival_time, concordance_index, order_grpah, mean_squared_error, hist_statistics, ErrHist
import json
from collections import defaultdict


before_steps = 5


if len(sys.argv) != 2:
    print "please add 1 parameter."
    exit(0)

if sys.argv[1] == "twitter":

    source_path = "../twitter/"
    X_train = np.load(source_path + "X_train_var.npy")
    T_train = np.load(source_path + "T_train_var.npy")
    C_train = np.load(source_path + "C_train_var.npy")

    X_test = np.load(source_path + "X_test_var.npy")
    T_test = np.load(source_path + "T_test_var.npy")
    C_test = np.load(source_path + "C_test_var.npy")

    X_valid = np.load(source_path + "X_valid_var.npy")
    T_valid = np.load(source_path + "T_valid_var.npy")
    C_valid = np.load(source_path + "C_valid_var.npy")

    # n_input = 5

elif sys.argv[1] == "wiki":

    source_path = "../wiki/v8/"
    X_train = np.load(source_path + "X_train.npy")
    T_train = np.load(source_path + "T_train.npy")
    C_train = np.load(source_path + "C_train.npy")

    X_test = np.load(source_path + "X_test.npy")
    T_test = np.load(source_path + "T_test.npy")
    C_test = np.load(source_path + "C_test.npy")

    X_valid = np.load(source_path + "X_valid.npy")
    T_valid = np.load(source_path + "T_valid.npy")
    C_valid = np.load(source_path + "C_valid.npy")

    # n_input = 8

else:
    print "parameter is not right, twitter or wiki."
    exit(0)


def reduce_to_cox_data_mean_twitter(X,T,C,steps):
    rec_coll = list()
    seq_c = list()
    labels = list()
    for _x, _t, _c in zip(X,T,C):
        rec_coll.append(np.mean(_x, axis=0).tolist() + [_t] + [_c])
        if _c == 0:
            seq_c.append(np.ones(steps))
            labels.append(1)
        elif _c == 1:
            seq_c.append(np.zeros(steps))
            labels.append(0)
    return df(rec_coll, columns=['follower_#', 'friend_#', 'member_#', 'favor_#', 'tweet_#', 'Survival', 'Event']), np.array(seq_c), np.array(labels)


def reduce_to_cox_data_steps_twitter(X,T,C,step,steps):
    rec_coll = list()
    seq_c = list()
    for _x, _t, _c in zip(X,T,C):
        rec_coll.append(_x[step-1].tolist() + [_t] + [_c])
        if _c == 0:
            seq_c.append(np.ones(steps))
        elif _c == 1:
            seq_c.append(np.zeros(steps))
    return df(rec_coll, columns=['follower_#', 'friend_#', 'member_#', 'favor_#', 'tweet_#', 'Survival', 'Event']), np.array(seq_c)



def reduce_to_cox_data_mean_wiki(X,T,C,steps):
    rec_coll = list()
    seq_c = list()
    labels = list()
    for _x, _t, _c in zip(X,T,C):
        rec_coll.append(np.mean(_x, axis=0).tolist() + [_t] + [_c])
        if _c == 0:
            seq_c.append(np.ones(steps))
            labels.append(1)
        elif _c == 1:
            seq_c.append(np.zeros(steps))
            labels.append(0)
    return df(rec_coll, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'Survival', 'Event']), np.array(seq_c), np.array(labels)


def reduce_to_cox_data_steps_wiki(X,T,C,step,steps):
    rec_coll = list()
    seq_c = list()
    for _x, _t, _c in zip(X,T,C):
        rec_coll.append(_x[step-1].tolist() + [_t] + [_c])
        if _c == 0:
            seq_c.append(np.ones(steps))
        elif _c == 1:
            seq_c.append(np.zeros(steps))
    return df(rec_coll, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'Survival', 'Event']), np.array(seq_c)


def svm_svc(X,y):
    clf = svm.SVC()
    return clf.fit(X,y)




if sys.argv[1] == "twitter":
    _X_train, _, yyy = reduce_to_cox_data_mean_twitter(X_train,T_train,C_train,before_steps)
else:
    _X_train, _, yyy = reduce_to_cox_data_mean_wiki(X_train, T_train, C_train, before_steps)

# _X_valid,  batch_y_valid = reduce_to_cox_data_mean(X_valid,T_valid,C_valid,before_steps)

if sys.argv[1] == "twitter":
    _X_valid_1,  batch_y_valid = reduce_to_cox_data_steps_twitter(X_valid,T_valid,C_valid,1,before_steps)
    _X_valid_2,  _ = reduce_to_cox_data_steps_twitter(X_valid,T_valid,C_valid,2,before_steps)
    _X_valid_3,  _ = reduce_to_cox_data_steps_twitter(X_valid,T_valid,C_valid,3,before_steps)
    _X_valid_4,  _ = reduce_to_cox_data_steps_twitter(X_valid,T_valid,C_valid,4,before_steps)
    _X_valid_5,  _ = reduce_to_cox_data_steps_twitter(X_valid,T_valid,C_valid,5,before_steps)
else:
    _X_valid_1,  batch_y_valid = reduce_to_cox_data_steps_wiki(X_valid,T_valid,C_valid,1,before_steps)
    _X_valid_2,  _ = reduce_to_cox_data_steps_wiki(X_valid,T_valid,C_valid,2,before_steps)
    _X_valid_3,  _ = reduce_to_cox_data_steps_wiki(X_valid,T_valid,C_valid,3,before_steps)
    _X_valid_4,  _ = reduce_to_cox_data_steps_wiki(X_valid,T_valid,C_valid,4,before_steps)
    _X_valid_5,  _ = reduce_to_cox_data_steps_wiki(X_valid,T_valid,C_valid,5,before_steps)

# fit the CoxPH model
cph = CoxPHFitter()
cph.fit(_X_train, duration_col='Survival', event_col='Event')
# cph.print_summary()  # access the results using cph.summary

# Validation
# _X_valid = _X_valid.drop(["Survival", "Event"], axis=1)  # Testing input after preprocessing.

_X_valid_1 = _X_valid_1.drop(["Survival", "Event"], axis=1)
_X_valid_2 = _X_valid_2.drop(["Survival", "Event"], axis=1)
_X_valid_3 = _X_valid_3.drop(["Survival", "Event"], axis=1)
_X_valid_4 = _X_valid_4.drop(["Survival", "Event"], axis=1)
_X_valid_5 = _X_valid_5.drop(["Survival", "Event"], axis=1)

# _seq_pred_y_valid_1 = cph.predict_survival_function(_X_valid_1, np.arange(before_steps)).as_matrix()

_seq_pred_y_valid = np.array([cph.predict_survival_function(_X_valid_1, np.arange(before_steps+1)).as_matrix()[1,:],
cph.predict_survival_function(_X_valid_2, np.arange(before_steps+1)).as_matrix()[2,:],
cph.predict_survival_function(_X_valid_3, np.arange(before_steps+1)).as_matrix()[3,:],
cph.predict_survival_function(_X_valid_4, np.arange(before_steps+1)).as_matrix()[4,:],
cph.predict_survival_function(_X_valid_5, np.arange(before_steps+1)).as_matrix()[5,:]])

_seq_pred_y_valid = _seq_pred_y_valid.transpose()

thrld_score = dict()
for sur_thrld_valid in np.arange(1.0, 0.0, -0.01):

    yy = []
    pp = []

    seq_pred_y_valid = np.zeros((_seq_pred_y_valid.shape[0], before_steps))
    seq_pred_y_valid[np.where(_seq_pred_y_valid[:, :before_steps] > sur_thrld_valid)] = 1

    early_correct = np.sum(seq_pred_y_valid == batch_y_valid, axis=0)
    yy.extend(batch_y_valid)
    pp.extend(seq_pred_y_valid)

    yy = np.asarray(yy)
    pp = np.asarray(pp)
    gt = np.logical_not(yy).astype(float)
    pr = np.logical_not(pp).astype(float)
    _precision, _recall, _F1 = prec_reca_F1(gt, pr)

    seq_corr_rate = early_correct/float(_seq_pred_y_valid.shape[0])
    thrld_score[sur_thrld_valid] = np.mean(seq_corr_rate)

best_thrld, acc_best_threshold = max(thrld_score.iteritems(), key=operator.itemgetter(1))
# print best_thrld, acc_best_threshold, seq_corr_rate
# # exit(0)

# best_thrld = 0.9505
#
# print "-----", best_thrld

yy = []
pp = []

# before_steps2 = 5


if sys.argv[1] == "twitter":
    _X_test_1,  batch_y_test = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,1,before_steps)
    _X_test_2,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,2,before_steps)
    _X_test_3,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,3,before_steps)
    _X_test_4,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,4,before_steps)
    _X_test_5,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,5,before_steps)
else:
    _X_test_1,  batch_y_test = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,1,before_steps)
    _X_test_2,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,2,before_steps)
    _X_test_3,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,3,before_steps)
    _X_test_4,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,4,before_steps)
    _X_test_5,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,5,before_steps)

# _X_test,  batch_y_test = reduce_to_cox_data_mean(X_test,T_test,C_test,before_steps2)

_X_test_1 = _X_test_1.drop(["Survival", "Event"], axis=1)
_X_test_2 = _X_test_2.drop(["Survival", "Event"], axis=1)
_X_test_3 = _X_test_3.drop(["Survival", "Event"], axis=1)
_X_test_4 = _X_test_4.drop(["Survival", "Event"], axis=1)
_X_test_5 = _X_test_5.drop(["Survival", "Event"], axis=1)

# _X_test = _X_test.drop(["Survival", "Event"], axis=1)  # Testing input after preprocessing.
# _seq_pred_y_test = cph.predict_survival_function(_X_test, np.arange(10)).transpose().as_matrix()

_seq_pred_y_test = np.array([cph.predict_survival_function(_X_test_1, np.arange(before_steps+1)).as_matrix()[1,:],
cph.predict_survival_function(_X_test_2, np.arange(before_steps+1)).as_matrix()[2,:],
cph.predict_survival_function(_X_test_3, np.arange(before_steps+1)).as_matrix()[3,:],
cph.predict_survival_function(_X_test_4, np.arange(before_steps+1)).as_matrix()[4,:],
cph.predict_survival_function(_X_test_5, np.arange(before_steps+1)).as_matrix()[5,:]])


_seq_pred_y_test = _seq_pred_y_test.transpose()

seq_pred_y_test = np.zeros((_seq_pred_y_test.shape[0], before_steps))
seq_pred_y_test[np.where(_seq_pred_y_test[:, :before_steps] > best_thrld)] = 1

early_correct = np.sum(seq_pred_y_test == batch_y_test, axis=0)

yy.extend(batch_y_test)
pp.extend(seq_pred_y_test)

yy = np.asarray(yy)
pp = np.asarray(pp)
gt = np.logical_not(yy).astype(float)
pr = np.logical_not(pp).astype(float)
_precision, _recall, _F1 = prec_reca_F1(gt, pr)

seq_corr_rate = early_correct/float(_seq_pred_y_test.shape[0])
# seq_corr_rate = np.sum(np.asarray(early_correct), axis=0)/float(_seq_pred_y_test.shape[0])

print "Cox:"
print "precision: ", _precision
print "recall: ", _recall
print "F1: ", _F1
print "accuracy: ", seq_corr_rate


# SVM-Survival analysis
if sys.argv[1] == "twitter":
    _X_test_1,  batch_y_test = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,1,6)
    _X_test_2,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,2,before_steps)
    _X_test_3,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,3,before_steps)
    _X_test_4,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,4,before_steps)
    _X_test_5,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,5,before_steps)
    _X_test_6,  _ = reduce_to_cox_data_steps_twitter(X_test,T_test,C_test,6,before_steps)
else:
    _X_test_1,  batch_y_test = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,1,6)
    _X_test_2,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,2,before_steps)
    _X_test_3,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,3,before_steps)
    _X_test_4,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,4,before_steps)
    _X_test_5,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,5,before_steps)
    _X_test_6,  _ = reduce_to_cox_data_steps_wiki(X_test,T_test,C_test,6,before_steps)

_X_test_1 = _X_test_1.drop(["Survival", "Event"], axis=1)
_X_test_2 = _X_test_2.drop(["Survival", "Event"], axis=1)
_X_test_3 = _X_test_3.drop(["Survival", "Event"], axis=1)
_X_test_4 = _X_test_4.drop(["Survival", "Event"], axis=1)
_X_test_5 = _X_test_5.drop(["Survival", "Event"], axis=1)
_X_test_6 = _X_test_6.drop(["Survival", "Event"], axis=1)


XXX = _X_train.drop(["Survival", "Event"], axis=1)
# yyy = np.logical_not(C_train)
svm_clf = svm_svc(XXX,yyy)

seq_pred_y_valid = np.array([svm_clf.predict(_X_valid_1),
                    svm_clf.predict(_X_valid_2),
                    svm_clf.predict(_X_valid_3),
                    svm_clf.predict(_X_valid_4),
                    svm_clf.predict(_X_valid_5)])

yy = []
pp = []

seq_pred_y_valid = seq_pred_y_valid.transpose()

early_correct = np.sum(seq_pred_y_valid == batch_y_valid, axis=0)

yy.extend(batch_y_valid)
pp.extend(seq_pred_y_valid)

yy = np.asarray(yy)
pp = np.asarray(pp)
gt = np.logical_not(yy).astype(float)
pr = np.logical_not(pp).astype(float)
_precision, _recall, _F1 = prec_reca_F1(gt, pr)

seq_corr_rate = early_correct/float(seq_pred_y_valid.shape[0])
print "\n\n"
print "SVM:"
print "precision: ", _precision
print "recall: ", _recall
print "F1: ", _F1
print "accuracy: ", seq_corr_rate

exit(0)
















































































































# exit(0)

#
# print "---------------------"
#
#
# # fit the CoxPH model
# cph = CoxPHFitter()
# cph.fit(twitter_dataset_tr, duration_col='Survival', event_col='Event')
# # cph.print_summary()  # access the results using cph.summary
#
# # evaluate the model
# X_test = twitter_dataset_ts.drop(["Survival", "Event"], axis=1)  # Testing input after preprocessing.
# sur_pro = cph.predict_survival_function(X_test).transpose().as_matrix()
#
# print sur_pro.shape
# print sur_pro[1]
# exit(0)
# #
# # obtain survival time for each user.
#
# thrld = 0.5          # Until one time stamp is less than this threshold,  survival time will be marked as its previous one.
#
# time_stamp = [0,3,4,11,13,18,20,22]
# f_sur_tim = get_survival_time(sur_pro, thrld, time_stamp, usr_ts)
#
#
# # get order_graph based on our observation.
#
# uncen_usr = json.load(open("./Data/uncen_usr.json", "r"))
# cen_usr_las_tim = json.load(open("./Data/cen_usr_las_tim.json", "r"))
# cen_usr_mis = json.load(open("./Data/cen_usr_mis.json", "r"))
#
# ord_gra = order_grpah(uncen_usr, cen_usr_las_tim, cen_usr_mis)
#
#
# # for usr in list(set(uncen_usr.keys()).intersection(f_sur_tim.keys())):
# #     if uncen_usr[usr] == 3:
# #         print f_sur_tim[usr]
#
# # exit(0)
# # CI
# CI, count, edge_n = concordance_index(ord_gra, f_sur_tim, False, 0)
#
# # print CI, count, edge_n
# # exit(0)
#
# CI_20 = concordance_index(ord_gra, f_sur_tim, True, 20)
# print "CI: ", CI
# print "CI_20: ", CI_20
#
# # mse
# mea, std_der = mean_squared_error(uncen_usr, f_sur_tim, False, 0)
# mea_20, std_der_20 = mean_squared_error(uncen_usr, f_sur_tim, True, 20)
#
# print "mean, std:  (%s, %s)"%(mea, std_der)
# print "mean_20, std_20:  (%s, %s)"%(mea_20, std_der_20)
#
# hist_total, hist_uncen_usr, hist_cen_usr_las_tim, hist_cen_usr_mis = hist_statistics(uncen_usr, cen_usr_las_tim, cen_usr_mis)
#
# ErrHist(hist_uncen_usr, hist_cen_usr_las_tim, hist_cen_usr_mis)
#
#
# exit(0)