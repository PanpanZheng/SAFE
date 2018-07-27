import numpy as np
from collections import defaultdict
from safdKit import prec_reca_F1




x = [[0,1,1,0],[1,1,1,0],[0,1,1,1],[0,0,1,0], [0,0,1,0]]

y = [[1,0,1,0],[0,0,0,0],[0,1,1,1],[1,0,1,0], [1,1,0,1]]

x = np.array(x)
y = np.array(y)




print x

print
print
print

print y

precision, recall, F1 = prec_reca_F1(x,y)

print
print

print "precison: ", precision
print

print "recall: ", recall
print

print "F1: ", F1
print

exit(0)




def get_first_beat(x, y):

    x_1 = np.where(x==y)[0]
    x_2 = np.where(x==y)[1]

    # print x_1
    # print x_2
    # exit(0)
    d = defaultdict(list)
    for _x_1, _x_2 in zip(x_1, x_2):
        d[_x_1].append(_x_2)

    first_beat = list()
    for k, v in d.items():
        first_beat.append(v[0])
    first_beat = np.array(first_beat) + 1
    return first_beat


# print x
#
# print "-------------------"
#
# print y
#
# print "\n"
print get_first_beat(x, y)

# gt = np.array([4,4,4,4])
# print gt-get_first_beat(x, y)
























exit(0)








source_path = "../diff_data_2/"

X_train = np.load(source_path + "X_train_var.npy")
T_train = np.load(source_path + "T_train_var.npy")
C_train = np.load(source_path + "C_train_var.npy")

X_valid = np.load(source_path + "X_valid_var.npy")
T_valid = np.load(source_path + "T_valid_var.npy")
C_valid = np.load(source_path + "C_valid_var.npy")

X_test = np.load(source_path + "X_test_var.npy")
T_test = np.load(source_path + "T_test_var.npy")
C_test = np.load(source_path + "C_test_var.npy")

print X_train.shape, T_train.shape, C_train.shape
print X_valid.shape, T_valid.shape, C_valid.shape
print X_test.shape, T_test.shape, C_test.shape

exit(0)


# source_path = "../wiki/"
#
# X_train = np.load(source_path + "X_train.npy")
# T_train = np.load(source_path + "T_train.npy")
# C_train = np.load(source_path + "C_train.npy")
#
# X_test = np.load(source_path + "X_test.npy")
# T_test = np.load(source_path + "T_test.npy")
# C_test = np.load(source_path + "C_test.npy")
#
# X_valid = np.load(source_path + "X_valid.npy")
# T_valid = np.load(source_path + "T_valid.npy")
# C_valid = np.load(source_path + "C_valid.npy")

print
print "Tranining: \n"

coll_train = []
count = 0
for i in np.unique(T_train):
    print i, ": ", np.sum(T_train == i)
    count += np.sum(T_train == i)
    coll_train.append(np.sum(T_train == i))

print "Total: ", count


print
print "Validataion: \n"

coll_valid = []
count = 0
for i in np.unique(T_valid):
    print i, ": ", np.sum(T_valid == i)
    count += np.sum(T_valid == i)
    coll_valid.append(np.sum(T_valid == i))

print "Total: ", count


print
print "Testing: \n"

coll_test = []
count = 0
for i in np.unique(T_test):
    print i, ": ", np.sum(T_test == i)
    count += np.sum(T_test == i)
    coll_test.append(np.sum(T_test == i))

print "Total: ", count


print "Censor/Uncensor: ", np.sum(C_train == 0), np.sum(C_train == 1), np.sum(C_valid == 0), np.sum(C_valid == 1), np.sum(C_test == 0), np.sum(C_test == 1)


coll_train = np.array(coll_train)
coll_valid = np.array(coll_valid)
coll_test = np.array(coll_test)

print coll_train + coll_valid + coll_test


exit(0)