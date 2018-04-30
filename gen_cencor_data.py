import numpy as np
import json
from collections import defaultdict
from safdKit import sample_shuffle_uspv


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


uncen_usr_data = defaultdict(list)
cen_usr_las_tim_data = defaultdict(list)
cen_usr_mis_data = defaultdict(list)

for usr in uncen_usr.keys():
    uncen_usr_data[usr]\
        = [followers_count[usr][0], friends_count[usr][0],  listed_count[usr][0],
           favourites_count[usr][0],statuses_count[usr][0], uncen_usr[usr], 1]

for usr in cen_usr_las_tim.keys():
    cen_usr_las_tim_data[usr]\
        = [followers_count[usr][0], friends_count[usr][0],  listed_count[usr][0],
           favourites_count[usr][0],statuses_count[usr][0], cen_usr_las_tim[usr], 0]

for usr in cen_usr_mis.keys():
    cen_usr_mis_data[usr]\
        = [followers_count[usr][0], friends_count[usr][0],  listed_count[usr][0],
           favourites_count[usr][0],statuses_count[usr][0], cen_usr_mis[usr], 0]


with open(dest_path + 'uncen_usr_data.json', 'w') as f:
    json.dump(uncen_usr_data, f)

with open(dest_path + 'cen_usr_las_tim_data.json', 'w') as f:
    json.dump(cen_usr_las_tim_data, f)

with open(dest_path + 'cen_usr_mis_data.json', 'w') as f:
    json.dump(cen_usr_mis_data, f)


exit(0)