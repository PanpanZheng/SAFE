import numpy as np
import sys
sys.path.append("../")
from safdKit import k_NN, decision_tree, random_forest, svm_svc, ran_seed
from sklearn.metrics import accuracy_score


path = "../diff_data/"
dest = "../diff_data_2/"

X_train_diff = np.load(path + "X_train_diff.npy")
T_train_diff = np.load(path + "T_train_diff.npy")
C_train_diff = np.load(path + "C_train_diff.npy")

X_test_diff = np.load(path + "X_test_diff.npy")
T_test_diff = np.load(path + "T_test_diff.npy")
C_test_diff = np.load(path + "C_test_diff.npy")

X_valid_diff = np.load(path + "X_valid_diff.npy")
T_valid_diff = np.load(path + "T_valid_diff.npy")
C_valid_diff = np.load(path + "C_valid_diff.npy")

X_train_21 = X_train_diff[T_train_diff==21].tolist()

X_train_20 = X_train_diff[T_train_diff==20].tolist()[:68]

X_train_19 = X_train_diff[T_train_diff==19].tolist()
X_train_18 = X_train_diff[T_train_diff==18].tolist()
X_train_17 = X_train_diff[T_train_diff==17].tolist()

X_train_16 = X_train_diff[T_train_diff==16].tolist()
X_train_15 = X_train_diff[T_train_diff==15].tolist()
X_train_14 = X_train_diff[T_train_diff==14].tolist()
X_train_13 = X_train_diff[T_train_diff==13].tolist()

X_train_12 = X_train_diff[T_train_diff==12].tolist()

X_test_21 = X_test_diff[T_test_diff==21].tolist()

X_test_20 = X_test_diff[T_test_diff==20].tolist()[:19]

X_test_19 = X_test_diff[T_test_diff==19].tolist()
X_test_18 = X_test_diff[T_test_diff==18].tolist()
X_test_17 = X_test_diff[T_test_diff==17].tolist()



X_test_16 = X_test_diff[T_test_diff==16].tolist()
X_test_15 = X_test_diff[T_test_diff==15].tolist()
X_test_14 = X_test_diff[T_test_diff==14].tolist()
X_test_13 = X_test_diff[T_test_diff==13].tolist()


X_test_12 = X_test_diff[T_test_diff==12].tolist()

X_valid_21 = X_valid_diff[T_valid_diff==21].tolist()

X_valid_20 = X_valid_diff[T_valid_diff==20].tolist()[:10]

X_valid_19 = X_valid_diff[T_valid_diff==19].tolist()
X_valid_18 = X_valid_diff[T_valid_diff==18].tolist()
X_valid_17 = X_valid_diff[T_valid_diff==17].tolist()


X_valid_16 = X_valid_diff[T_valid_diff==16].tolist()
X_valid_15 = X_valid_diff[T_valid_diff==15].tolist()
X_valid_14 = X_valid_diff[T_valid_diff==14].tolist()
X_valid_13 = X_valid_diff[T_valid_diff==13].tolist()

X_valid_12 = X_valid_diff[T_valid_diff==12].tolist()


X_train_21, X_train_19, X_train_18, X_train_17, X_train_12 = np.array(X_train_21), np.array(X_train_19), np.array(X_train_18), np.array(X_train_17), np.array(X_train_12)
X_test_21, X_test_19, X_test_18, X_test_17, X_test_12 =  np.array(X_test_21), np.array(X_test_19), np.array(X_test_18), np.array(X_test_17), np.array(X_test_12)
X_valid_21, X_valid_19, X_valid_18, X_valid_17, X_valid_12 = np.array(X_valid_21), np.array(X_valid_19), np.array(X_valid_18), np.array(X_valid_17), np.array(X_valid_12)

X_train_19 = X_train_19[0:int((len(X_train_18) + len(X_train_17) + len(X_train_12))/3)]
X_test_19 = X_test_19[0:int((len(X_test_18) + len(X_test_17) + len(X_test_12))/3)]
X_valid_19 = X_valid_19[0:int((len(X_valid_18) + len(X_valid_17) + len(X_valid_12))/3)]

X_train_20, X_train_16, X_train_15, X_train_14, X_train_13 = np.array(X_train_20), np.array(X_train_16), \
                                                             np.array(X_train_15), np.array(X_train_14), np.array(X_train_13)

X_test_20, X_test_16, X_test_15, X_test_14, X_test_13 = np.array(X_test_20), np.array(X_test_16), \
                                                             np.array(X_test_15), np.array(X_test_14), np.array(X_test_13)

X_valid_20, X_valid_16, X_valid_15, X_valid_14, X_valid_13 = np.array(X_valid_20), np.array(X_valid_16), \
                                                             np.array(X_valid_15), np.array(X_valid_14), np.array(X_valid_13)


s1 = len(X_train_19) + len(X_train_18) + len(X_train_17) + len(X_train_12) + len(X_train_20) + len(X_train_16) + len(X_train_15) + len(X_train_14) + len(X_train_13)

s2 = len(X_test_19) + len(X_test_18) + len(X_test_17) + len(X_test_12)+ len(X_test_20) + len(X_test_16) + len(X_test_15) + len(X_test_14) + len(X_test_13)

s3 = len(X_valid_19) +  len(X_valid_18) +  len(X_valid_17) +  len(X_valid_12) + len(X_valid_20) + len(X_valid_16) + len(X_valid_15) + len(X_valid_14) + len(X_valid_13)

s = ran_seed(X_train_21.shape[0])
X_train_21 = (X_train_21[s])[0:s1]

s = ran_seed(X_test_21.shape[0])
X_test_21 = (X_test_21[s])[0:s2]

s = ran_seed(X_valid_21.shape[0])
X_valid_21 = (X_valid_21[s])[0:s3]


X_train = list()

X_train.extend(X_train_12)

X_train.extend(X_train_13)
X_train.extend(X_train_14)
X_train.extend(X_train_15)
X_train.extend(X_train_16)

X_train.extend(X_train_17)
X_train.extend(X_train_18)
X_train.extend(X_train_19)

X_train.extend(X_train_20)

X_train.extend(X_train_21)

T_train = list()
T_train.extend((np.zeros(X_train_12.shape[0])+12).tolist())

T_train.extend((np.zeros(X_train_13.shape[0])+13).tolist())
T_train.extend((np.zeros(X_train_14.shape[0])+14).tolist())
T_train.extend((np.zeros(X_train_15.shape[0])+15).tolist())
T_train.extend((np.zeros(X_train_16.shape[0])+16).tolist())

T_train.extend((np.zeros(X_train_17.shape[0])+17).tolist())
T_train.extend((np.zeros(X_train_18.shape[0])+18).tolist())
T_train.extend((np.zeros(X_train_19.shape[0])+19).tolist())

T_train.extend((np.zeros(X_train_20.shape[0])+20).tolist())

T_train.extend((np.zeros(X_train_21.shape[0])+21).tolist())


C_train = list()

C_train.extend((np.ones(X_train_12.shape[0])).tolist())

C_train.extend((np.ones(X_train_13.shape[0])).tolist())
C_train.extend((np.ones(X_train_14.shape[0])).tolist())
C_train.extend((np.ones(X_train_15.shape[0])).tolist())
C_train.extend((np.ones(X_train_16.shape[0])).tolist())

C_train.extend((np.ones(X_train_17.shape[0])).tolist())
C_train.extend((np.ones(X_train_18.shape[0])).tolist())
C_train.extend((np.ones(X_train_19.shape[0])).tolist())

C_train.extend((np.ones(X_train_20.shape[0])).tolist())

C_train.extend((np.zeros(X_train_21.shape[0])).tolist())


X_train, T_train, C_train = np.array(X_train), np.array(T_train), np.array(C_train)

# print X_train.shape, T_train.shape, C_train.shape
# exit(0)

np.save(dest + "X_train_var", X_train)
np.save(dest + "T_train_var", T_train)
np.save(dest + "C_train_var", C_train)


X_test = list()

X_test.extend(X_test_12)

X_test.extend(X_test_13)
X_test.extend(X_test_14)
X_test.extend(X_test_15)
X_test.extend(X_test_16)

X_test.extend(X_test_17)
X_test.extend(X_test_18)
X_test.extend(X_test_19)

X_test.extend(X_test_20)

X_test.extend(X_test_21)


T_test = list()
T_test.extend((np.zeros(X_test_12.shape[0])+12).tolist())

T_test.extend((np.zeros(X_test_13.shape[0])+13).tolist())
T_test.extend((np.zeros(X_test_14.shape[0])+14).tolist())
T_test.extend((np.zeros(X_test_15.shape[0])+15).tolist())
T_test.extend((np.zeros(X_test_16.shape[0])+16).tolist())

T_test.extend((np.zeros(X_test_17.shape[0])+17).tolist())
T_test.extend((np.zeros(X_test_18.shape[0])+18).tolist())
T_test.extend((np.zeros(X_test_19.shape[0])+19).tolist())

T_test.extend((np.zeros(X_test_20.shape[0])+20).tolist())

T_test.extend((np.zeros(X_test_21.shape[0])+21).tolist())

C_test = list()
C_test.extend((np.ones(X_test_12.shape[0])).tolist())

C_test.extend((np.ones(X_test_13.shape[0])).tolist())
C_test.extend((np.ones(X_test_14.shape[0])).tolist())
C_test.extend((np.ones(X_test_15.shape[0])).tolist())
C_test.extend((np.ones(X_test_16.shape[0])).tolist())

C_test.extend((np.ones(X_test_17.shape[0])).tolist())
C_test.extend((np.ones(X_test_18.shape[0])).tolist())
C_test.extend((np.ones(X_test_19.shape[0])).tolist())

C_test.extend((np.ones(X_test_20.shape[0])).tolist())

C_test.extend((np.zeros(X_test_21.shape[0])).tolist())

X_test, T_test, C_test = np.array(X_test), np.array(T_test), np.array(C_test)

# print X_test.shape, T_test.shape, C_test.shape
# exit(0)

np.save(dest + "X_test_var", X_test)
np.save(dest + "T_test_var", T_test)
np.save(dest + "C_test_var", C_test)

X_valid = list()

X_valid.extend(X_valid_12)

X_valid.extend(X_valid_13)
X_valid.extend(X_valid_14)
X_valid.extend(X_valid_15)
X_valid.extend(X_valid_16)

X_valid.extend(X_valid_17)
X_valid.extend(X_valid_18)
X_valid.extend(X_valid_19)

X_valid.extend(X_valid_20)

X_valid.extend(X_valid_21)


T_valid = list()
T_valid.extend((np.zeros(X_valid_12.shape[0])+12).tolist())

T_valid.extend((np.zeros(X_valid_13.shape[0])+13).tolist())
T_valid.extend((np.zeros(X_valid_14.shape[0])+14).tolist())
T_valid.extend((np.zeros(X_valid_15.shape[0])+15).tolist())
T_valid.extend((np.zeros(X_valid_16.shape[0])+16).tolist())

T_valid.extend((np.zeros(X_valid_17.shape[0])+17).tolist())
T_valid.extend((np.zeros(X_valid_18.shape[0])+18).tolist())
T_valid.extend((np.zeros(X_valid_19.shape[0])+19).tolist())

T_valid.extend((np.zeros(X_valid_20.shape[0])+20).tolist())

T_valid.extend((np.zeros(X_valid_21.shape[0])+21).tolist())

C_valid = list()
C_valid.extend((np.ones(X_valid_12.shape[0])).tolist())

C_valid.extend((np.ones(X_valid_13.shape[0])).tolist())
C_valid.extend((np.ones(X_valid_14.shape[0])).tolist())
C_valid.extend((np.ones(X_valid_15.shape[0])).tolist())
C_valid.extend((np.ones(X_valid_16.shape[0])).tolist())

C_valid.extend((np.ones(X_valid_17.shape[0])).tolist())
C_valid.extend((np.ones(X_valid_18.shape[0])).tolist())
C_valid.extend((np.ones(X_valid_19.shape[0])).tolist())

C_valid.extend((np.ones(X_valid_20.shape[0])).tolist())

C_valid.extend((np.zeros(X_valid_21.shape[0])).tolist())

X_valid, T_valid, C_valid = np.array(X_valid), np.array(T_valid), np.array(C_valid)

# print X_valid.shape, T_valid.shape, C_valid.shape
# exit(0)
np.save(dest + "X_valid_var", X_valid)
np.save(dest + "T_valid_var", T_valid)
np.save(dest + "C_valid_var", C_valid)


exit(0)












# source_path = "../Data2/"
# dest_path = "../Data4/"
#
# X_train_diff = np.load(source_path + "X_train_diff.npy")
# T_train_diff = np.load(source_path + "T_train_diff.npy")
# C_train_diff = np.load(source_path + "C_train_diff.npy")
# #
# X_test_diff = np.load(source_path + "X_test_diff.npy")
# T_test_diff = np.load(source_path + "T_test_diff.npy")
# C_test_diff = np.load(source_path + "C_test_diff.npy")


# print X_train_diff.shape, T_train_diff.shape, C_train_diff.shape
# print X_test_diff.shape, T_test_diff.shape, C_test_diff.shape
#
# exit(0)
#
# D_train_22 = []
# # for x_seq in X_train_np[T_train==22].tolist():
# for x_seq in X_train_diff[T_train_diff==21].tolist():
#     tmp = []
#     for x in x_seq:
#         for e in x:
#             tmp.append(e)
#     D_train_22.append(tmp)
# D_train_22 = np.array(D_train_22)
#
# D_train_21 = []
# # for x_seq in X_train_np[T_train==17].tolist():
# for x_seq in X_train_diff[T_train_diff == 1].tolist():
#     tmp = []
#     for x in x_seq:
#         for e in x:
#             tmp.append(e)
#     D_train_21.append(tmp)
# D_train_21 = np.array(D_train_21)
#
# print D_train_22.shape, D_train_21.shape
#
# D_train = np.concatenate((D_train_22[0:100, -50:], D_train_21[:, -50:]), axis=0)
# y_train = np.concatenate((np.zeros(100), np.zeros(100)+1), axis=0)
#
# # D_train = np.concatenate((D_train_22[0:45, -50:], D_train_21[:, -50:]), axis=0)
# # y_train = np.concatenate((np.zeros(45), np.zeros(45)+1), axis=0)
#
#
# D_test_22 = []
# # for x_seq in X_test_np[T_test==22].tolist():
# for x_seq in X_test_diff[T_test_diff == 21].tolist():
#     tmp = []
#     for x in x_seq:
#         for e in x:
#             tmp.append(e)
#     D_test_22.append(tmp)
# D_test_22 = np.array(D_test_22)
#
# D_test_21 = []
# # for x_seq in X_test_np[T_test==17].tolist():
# for x_seq in X_test_diff[T_test_diff == 1].tolist():
#     tmp = []
#     for x in x_seq:
#         for e in x:
#             tmp.append(e)
#     D_test_21.append(tmp)
# D_test_21 = np.array(D_test_21)
#
# print D_test_22.shape, D_test_21.shape
# exit(0)

#
# D_test = np.concatenate((D_test_22[0:60, -50:], D_test_21[:, -50:]), axis=0)
# y_test = np.concatenate((np.zeros(60), np.zeros(60)+1), axis=0)
#
# #
# # print D_train.shape, y_train.shape, D_test.shape, y_test.shape
# # exit(0)
#
#
# clf_knn = k_NN(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "KNN: ", acc
#
#
# clf_knn = decision_tree(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "Decision_tree: ", acc
#
#
#
# clf_knn = random_forest(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "Random_forest: ", acc
#
#
# clf_knn = svm_svc(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "Svm_svc: ", acc
#
# # exit(0)
# np.save(source_path + "D_train_10", D_train)
# np.save(source_path + "y_train_10", y_train)
# np.save(source_path + "D_test_10", D_test)
# np.save(source_path + "y_test_10", y_test)
#
#
# exit(0)



# D_train_10 = np.load(source_path + "D_train_10.npy")
# y_train_10 = np.load(source_path + "y_train_10.npy")
# D_test_10 = np.load(source_path + "D_test_10.npy")
# y_test_10 = np.load(source_path + "y_test_10.npy")
#
# D_train_15 = np.load(source_path + "D_train_15.npy")
# y_train_15 = np.load(source_path + "y_train_15.npy")
# D_test_15 = np.load(source_path + "D_test_15.npy")
# y_test_15 = np.load(source_path + "y_test_15.npy")
#
# D_train_16 = np.load(source_path + "D_train_16.npy")
# y_train_16 = np.load(source_path + "y_train_16.npy")
# D_test_16 = np.load(source_path + "D_test_16.npy")
# y_test_16 = np.load(source_path + "y_test_16.npy")
#
# D_train_20 = np.load(source_path + "D_train_20.npy")
# y_train_20 = np.load(source_path + "y_train_20.npy")
# D_test_20 = np.load(source_path + "D_test_20.npy")
# y_test_20 = np.load(source_path + "y_test_20.npy")
#
#
# D_train = np.concatenate((D_train_10, D_train_15, D_train_16, D_train_20))
# y_train = np.concatenate((y_train_10, y_train_15, y_train_16, y_train_20))
# D_test = np.concatenate((D_test_10, D_test_15, D_test_16, D_test_20))
# y_test = np.concatenate((y_test_10, y_test_15, y_test_16, y_test_20))
#
# # print D_train.shape, y_train.shape, D_test.shape, y_test.shape
#
# clf_knn = k_NN(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "KNN: ", acc
#
#
# clf_knn = decision_tree(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "Decision_tree: ", acc
#
#
#
# clf_knn = random_forest(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "Random_forest: ", acc
#
#
# clf_knn = svm_svc(D_train,y_train)
# y_pred = clf_knn.predict(D_test)
# acc = accuracy_score(y_test, y_pred)
# print "Svm_svc: ", acc
#
#
# exit(0)