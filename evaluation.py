import warnings
warnings.filterwarnings("ignore")

from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from safdKit import load_twitter, get_survival_time, concordance_index, order_grpah, mean_squared_error, hist_statistics, ErrHist
import json


# load data
twitter_dataset_tr, twitter_dataset_ts, usr_ts = load_twitter()  # usr_ts is the set of users in testing.

# fit the CoxPH model
cph = CoxPHFitter()
cph.fit(twitter_dataset_tr, duration_col='Survival', event_col='Event')
# cph.print_summary()  # access the results using cph.summary

# evaluate the model
X_test = twitter_dataset_ts.drop(["Survival", "Event"], axis=1)  # Testing input after preprocessing.
sur_pro = cph.predict_survival_function(X_test).transpose().as_matrix()

# obtain survival time for each user.

thrld = 0.5          # Until one time stamp is less than this threshold,  survival time will be marked as its previous one.

time_stamp = [0,3,4,11,13,18,20,22]
f_sur_tim = get_survival_time(sur_pro, thrld, time_stamp, usr_ts)



# get order_graph based on our observation.

uncen_usr = json.load(open("./Data/uncen_usr.json", "r"))
cen_usr_las_tim = json.load(open("./Data/cen_usr_las_tim.json", "r"))
cen_usr_mis = json.load(open("./Data/cen_usr_mis.json", "r"))

ord_gra = order_grpah(uncen_usr, cen_usr_las_tim, cen_usr_mis)


# for usr in list(set(uncen_usr.keys()).intersection(f_sur_tim.keys())):
#     if uncen_usr[usr] == 3:
#         print f_sur_tim[usr]
#
# exit(0)
# CI
CI, count, edge_n = concordance_index(ord_gra, f_sur_tim, False, 0)

print CI, count, edge_n
exit(0)

CI_20 = concordance_index(ord_gra, f_sur_tim, True, 20)
print "CI: ", CI
print "CI_20: ", CI_20

# mse
mea, std_der = mean_squared_error(uncen_usr, f_sur_tim, False, 0)
mea_20, std_der_20 = mean_squared_error(uncen_usr, f_sur_tim, True, 20)

print "mean, std:  (%s, %s)"%(mea, std_der)
print "mean_20, std_20:  (%s, %s)"%(mea_20, std_der_20)

hist_total, hist_uncen_usr, hist_cen_usr_las_tim, hist_cen_usr_mis = hist_statistics(uncen_usr, cen_usr_las_tim, cen_usr_mis)

ErrHist(hist_uncen_usr, hist_cen_usr_las_tim, hist_cen_usr_mis)


exit(0)