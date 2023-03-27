# Bogdan-Marian Raia, Gr. 411 (DS)
# Smartphone User Identification
# Voting model using a LR, SVM, RF, LGBM

import os
import csv
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from time import time
from math import trunc
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# TODO: Find a better alternative than feature position hardcoding
np.set_printoptions(suppress=True) # Remove scientific notation

start_time = time()
nr_classes = 20
nr_validation = 2000
nr_test = 5000
nr_train = 7000
nr_features = 61

nr_windows_features = 7
windows_size = 26
nr_rows = 130

train_features = {}
train_labels = {}
train_folder = 'train' # copied train/test folders into main directory, without the second layer of train/test folders
train_files = os.listdir(train_folder)   

valid_features = {}
valid_labels = {}

test_features = {}
test_labels = {}
test_folder = 'test'
test_files = os.listdir(test_folder)

valid_files = train_files[-nr_validation:]
train_files = train_files[:-nr_validation] # ignore validation files

# Read the labels from "train_labels.csv", put them in train_labels and valid_labels
with open("train_labels.csv") as fp:
    label_file = csv.reader(fp)
    data_read = [row for row in label_file]

for idx, row in enumerate(data_read[1:]):
    if idx < nr_train:
        train_labels[row[0]] = row[1]
    else:
        valid_labels[row[0]] = row[1]

for file in train_files:
    file_name = os.path.join(train_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = np.array([[float(x) for x in row] for row in reader])
        file = file[:-4] # grab file name without '.csv'
        train_features[file] = np.empty(nr_features + nr_windows_features*nr_rows//windows_size)
        train_features[file][0:3] = np.mean(data_read, axis = 0)
        train_features[file][3:6] = np.std(data_read, axis = 0)
        train_features[file][6:9] = np.median(data_read, axis = 0)
        train_features[file][9:12] = np.percentile(data_read, 25, axis = 0)
        train_features[file][12:15] = np.percentile(data_read, 75, axis = 0)
        train_features[file][15:18] = np.min(data_read, axis = 0)
        train_features[file][18:21] = np.max(data_read, axis = 0)
        train_features[file][21:24] = np.argmax(data_read, axis = 0)/len(data_read)
        train_features[file][24:27] = skew(data_read, axis = 0)
        train_features[file][27:30] = kurtosis(data_read, axis = 0)
        train_features[file][30:33] = np.max(data_read, axis=0) - np.min(data_read, axis = 0)
        
        diffs = np.array(abs(np.array(data_read[1]) - np.array(data_read[0])))
        diffs = diffs[np.newaxis]
        distance = np.array((data_read[0][0]**2 + data_read[0][1]**2 + data_read[0][2]**2) ** (1/2))
        distance = distance[np.newaxis]
        distance = np.append(distance, np.array((data_read[1][0]**2 + data_read[1][1]**2 + data_read[1][2]**2) ** (1/2)))
        flips = np.zeros(3)
        
        for idx, row in enumerate(data_read):
            if idx >= 2:
                diffs = np.append(diffs, [abs(np.array(data_read[idx]) - np.array(data_read[idx-1]))], axis = 0)
                distance = np.append(distance, (data_read[idx][0]**2 + data_read[idx][1]**2 + data_read[idx][2]**2) ** (1/2))
                old_trend = np.array(data_read[idx-1]) - np.array(data_read[idx-2]) <= 0
                new_trend = np.array(data_read[idx]) - np.array(data_read[idx-1]) >= 0
                flips += old_trend != new_trend
        
        train_features[file][33:36] = np.std(diffs, axis = 0)
        train_features[file][36:39] = np.mean(diffs, axis = 0)
        train_features[file][39:42] = np.percentile(diffs, 25, axis = 0)
        train_features[file][42:45] = np.percentile(diffs, 75, axis = 0)
        train_features[file][45:48] = np.max(diffs, axis = 0)
        train_features[file][48:51] = flips/(0.5*len(data_read))
        train_features[file][51] = np.mean(distance)
        train_features[file][52] = np.std(distance)
        train_features[file][53] = np.min(distance)
        train_features[file][54] = np.max(distance)
        
        counter = 55
        for i in range(nr_rows//windows_size):
            train_features[file][counter:counter+3] = np.mean(data_read[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 3
            train_features[file][counter:counter+3] = np.mean(diffs[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 3
            train_features[file][counter] = np.mean(distance[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 1   
            
        N = len(data_read) # number of sample points
        T = 1/100 # sample spacing
        t = np.linspace(0, N*T, N)
        f = data_read - np.mean(data_read, axis = 0)
        window = hamming(N)
        f = f*window[:, np.newaxis]
        ft = fft(f, axis = 0)
        freq = fftfreq(N,T)[:N//2]
        
        for row in range(0,3):
            index = {freq[idx]: 0.5*abs(ft[:,row][idx])/N for idx in range(0, len(freq))}
            sorted_values = sorted(list(index.values()), reverse=True)
            train_features[file][counter] = sorted_values[0]
            counter += 1
            train_features[file][counter] = list(index.keys())[list(index.values()).index(sorted_values[0])]
            counter += 1

for file in valid_files:
    file_name = os.path.join(train_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = np.array([[float(x) for x in row] for row in reader]) 
        file = file[:-4] # grab file name without '.csv'
        
        valid_features[file] = np.empty(nr_features + nr_windows_features*nr_rows//windows_size)
        valid_features[file][0:3] = np.mean(data_read, axis = 0)
        valid_features[file][3:6] = np.std(data_read, axis = 0)
        valid_features[file][6:9] = np.median(data_read, axis = 0)
        valid_features[file][9:12] = np.percentile(data_read, 25, axis = 0)
        valid_features[file][12:15] = np.percentile(data_read, 75, axis = 0)
        valid_features[file][15:18] = np.min(data_read, axis = 0)
        valid_features[file][18:21] = np.max(data_read, axis = 0)
        valid_features[file][21:24] = np.argmax(data_read, axis = 0)/len(data_read)
        valid_features[file][24:27] = skew(data_read, axis = 0)
        valid_features[file][27:30] = kurtosis(data_read, axis = 0)
        valid_features[file][30:33] = np.max(data_read, axis=0) - np.min(data_read, axis = 0)
        
        diffs = np.array(abs(np.array(data_read[1]) - np.array(data_read[0])))
        diffs = diffs[np.newaxis]
        distance = np.array((data_read[0][0]**2 + data_read[0][1]**2 + data_read[0][2]**2) ** (1/2))
        distance = distance[np.newaxis]
        distance = np.append(distance, np.array((data_read[1][0]**2 + data_read[1][1]**2 + data_read[1][2]**2) ** (1/2)))
        flips = np.zeros(3)
        
        for idx, row in enumerate(data_read):
            if idx >= 2:
                diffs = np.append(diffs, [abs(np.array(data_read[idx]) - np.array(data_read[idx-1]))], axis = 0)
                distance = np.append(distance, (data_read[idx][0]**2 + data_read[idx][1]**2 + data_read[idx][2]**2) ** (1/2))
                old_trend = np.array(data_read[idx-1]) - np.array(data_read[idx-2]) <= 0
                new_trend = np.array(data_read[idx]) - np.array(data_read[idx-1]) >= 0
                flips += old_trend != new_trend
        
        valid_features[file][33:36] = np.std(diffs, axis = 0)
        valid_features[file][36:39] = np.mean(diffs, axis = 0)
        valid_features[file][39:42] = np.percentile(diffs, 25, axis = 0)
        valid_features[file][42:45] = np.percentile(diffs, 75, axis = 0)
        valid_features[file][45:48] = np.max(diffs, axis = 0)
        valid_features[file][48:51] = flips/(0.5*len(data_read))
        valid_features[file][51] = np.mean(distance)
        valid_features[file][52] = np.std(distance)
        valid_features[file][53] = np.min(distance)
        valid_features[file][54] = np.max(distance)
        
        counter = 55
        for i in range(nr_rows//windows_size):
            valid_features[file][counter:counter+3] = np.mean(data_read[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 3
            valid_features[file][counter:counter+3] = np.mean(diffs[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 3
            valid_features[file][counter] = np.mean(distance[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 1
            
        N = len(data_read) # number of sample points
        T = 1/100 # sample spacing
        t = np.linspace(0, N*T, N)
        f = data_read - np.mean(data_read, axis = 0)
        window = hamming(N)
        f = f*window[:, np.newaxis]
        ft = fft(f, axis = 0)
        freq = fftfreq(N,T)[:N//2]
        
        for row in range(0,3):
            index = {freq[idx]: 0.5*abs(ft[:,row][idx])/N for idx in range(0, len(freq))}
            sorted_values = sorted(list(index.values()), reverse=True)
            valid_features[file][counter] = sorted_values[0]
            counter += 1
            valid_features[file][counter] = list(index.keys())[list(index.values()).index(sorted_values[0])]
            counter += 1 

LR = LogisticRegression(C = 1000, solver = 'newton-cg', max_iter = 1000)
# parameters = {'C': [0.10, 10, 1000], 'solver': ('sag', 'lbfgs', 'newton-cg'), 'max_iter': [100, 1000, 3000]}
# GridSearchLR =  GridSearchCV(LR, parameters, verbose = 3)
# GridSearchLR.fit(np.array(list(train_features.values()) + list(valid_features.values())), np.array(list(train_labels.values()) + list(valid_labels.values())))
# print(f"5-CV accuracy for SVM: {GridSearchLR.best_score_}")
# print(f"Best hyperparameters for SVM: {GridSearchLR.best_params_}")
LR.fit(np.array(list(train_features.values())),np.array(list(train_labels.values())))
predicted_labels = LR.predict(np.array(list(valid_features.values())))
print(f"Validation accuracy for LR: {accuracy_score(list(valid_labels.values()), predicted_labels)}")
# Best hyperparams are max_iter as large as possible, C=1000, solver  = 'newton-cg'

SVM = svm.SVC(C = 1000, kernel = 'rbf', probability=True)
# parameters = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ('linear', 'rbf')}
# GridSearchSVM =  GridSearchCV(SVM, parameters, verbose=3)
# GridSearchSVM.fit(np.array(list(train_features.values()) + list(valid_features.values())), np.array(list(train_labels.values()) + list(valid_labels.values())))
# print(f"5-CV accuracy for SVM: {GridSearchSVM.best_score_}")
# print(f"Best hyperparameters for SVM: {GridSearchSVM.best_params_}")
SVM.fit(np.array(list(train_features.values())),np.array(list(train_labels.values())))
predicted_labels = SVM.predict(np.array(list(valid_features.values())))
print(f"Validation accuracy for SVM: {accuracy_score(list(valid_labels.values()), predicted_labels)}")
# Best hyperparams are C=1000, kernel  = 'rbf'/linear'

RF = RandomForestClassifier(n_estimators = 180, criterion = 'log_loss', max_features = 'sqrt')
# parameters = {'n_estimators' : [120, 150, 180, 210, 240, 270, 300], 'criterion': ('gini', 'entropy', 'log_loss'), 'max_features' : ('sqrt', 'log2')}
# GridSearchRF =  GridSearchCV(RF, parameters, verbose=3)
# GridSearchRF.fit(np.array(list(train_features.values()) + list(valid_features.values())),np.array(list(train_labels.values()) + list(valid_labels.values())))
# print(f"5-CV accuracy for RF: {GridSearchRF.best_score_}")
# print(f"Best hyperparameters for RF: {GridSearchRF.best_params_}")
RF.fit(np.array(list(train_features.values())),np.array(list(train_labels.values())))
predicted_labels = RF.predict(np.array(list(valid_features.values())))
print(f"Validation accuracy for RF: {accuracy_score(list(valid_labels.values()), predicted_labels)}")
# Best hyperparams are n_estimators = 180/210, criterion = 'gini'/'log_loss'

# def objective(trial):
#     lgb_reg_alpha = trial.suggest_float('lambda_l1', 1e-8, 10.0)
#     lgb_reg_lambda = trial.suggest_float('lambda_l2', 1e-8, 10.0)
#     lgb_num_leaves = trial.suggest_int('num_leaves', 2, 400)
#     lgb_feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
#     lgb_bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)
#     lgb_bagging_freq = trial.suggest_int('bagging_freq', 0, 15)
#     lgb_min_child_samples = trial.suggest_int('min_child_samples', 1, 500)
#     LGBM = lgb.LGBMClassifier(objective='multiclass', num_class = 20, reg_alpha=lgb_reg_alpha, reg_lambda=lgb_reg_lambda, num_leaves=lgb_num_leaves, 
#                               feature_fraction=lgb_feature_fraction, bagging_fraction=lgb_bagging_fraction, bagging_freq=lgb_bagging_freq, min_child_samples=lgb_min_child_samples)
#     LGBM.fit(np.array(list(train_features.values())),np.array(list(train_labels.values())))
#     predicted_labels = LGBM.predict(np.array(list(valid_features.values()))) 
#     return accuracy_score(list(valid_labels.values()), predicted_labels)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=300)
# print(f"Number of finished trials: {len(study.trials)}")
# print("Best trial:")
# trial = study.best_trial
# print(f"Value: {trial.value}")
# print("Params:")
# for key, value in trial.params.items():
#     print(f"{key}: {value}")
    
LGBM = lgb.LGBMClassifier(objective='multiclass', num_class = 20, reg_alpha=0.3564041546293353,
                          reg_lambda=0.33200671782740687, num_leaves=380, feature_fraction=0.5576887727357542,
                          bagging_fraction=0.8418509256400377, bagging_freq=6, min_child_samples=166)
LGBM.fit(np.array(list(train_features.values())),np.array(list(train_labels.values())))
predicted_labels = LGBM.predict(np.array(list(valid_features.values())))
# LGBM_scores = cross_val_score(LGBM, np.array(list(train_features.values()) + list(valid_features.values())),np.array(list(train_labels.values()) + list(valid_labels.values())))
# print(f"5-cv accuracy for LGBM: {np.mean(LGBM_scores)}")
print(f"Validation accuracy for LGBM: {accuracy_score(list(valid_labels.values()), predicted_labels)}")

clf = VotingClassifier(estimators=[('RF', RF), ('LR', LR), ('SVM', SVM), ('LGBM', LGBM)], voting='soft')
clf.fit(np.array(list(train_features.values())),np.array(list(train_labels.values())))
predicted_labels = clf.predict(np.array(list(valid_features.values())))
# clf_scores = cross_val_score(clf, np.array(list(train_features.values()) + list(valid_features.values())),np.array(list(train_labels.values()) + list(valid_labels.values())))
# print(f"5-cv accuracy for voting classifier: {np.mean(clf_scores)}")
print(f"Validation accuracy for voting classifier: {accuracy_score(list(valid_labels.values()), predicted_labels)}")
cm = confusion_matrix(list(valid_labels.values()), predicted_labels)
cmp = ConfusionMatrixDisplay(cm, display_labels = np.arange(1, 21))
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax)    

clf2 = VotingClassifier(estimators=[('RF', RF), ('LR', LR), ('SVM', SVM), ('LGBM', LGBM)], voting='soft')
clf2.fit(np.array(list(train_features.values()) + list(valid_features.values())),np.array(list(train_labels.values()) + list(valid_labels.values())))

for file in test_files:
    file_name = os.path.join(test_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = np.array([[float(x) for x in row] for row in reader])
        file = file[:-4] # grab file name without '.csv'
        test_features[file] = np.empty(nr_features + nr_windows_features*nr_rows//windows_size)
        test_features[file][0:3] = np.mean(data_read, axis = 0)
        test_features[file][3:6] = np.std(data_read, axis = 0)
        test_features[file][6:9] = np.median(data_read, axis = 0)
        test_features[file][9:12] = np.percentile(data_read, 25, axis = 0)
        test_features[file][12:15] = np.percentile(data_read, 75, axis = 0)
        test_features[file][15:18] = np.min(data_read, axis = 0)
        test_features[file][18:21] = np.max(data_read, axis = 0)
        test_features[file][21:24] = np.argmax(data_read, axis = 0)/len(data_read)
        test_features[file][24:27] = skew(data_read, axis = 0)
        test_features[file][27:30] = kurtosis(data_read, axis = 0)
        test_features[file][30:33] = np.max(data_read, axis=0) - np.min(data_read, axis = 0)
        
        diffs = np.array(abs(np.array(data_read[1]) - np.array(data_read[0])))
        diffs = diffs[np.newaxis]
        distance = np.array((data_read[0][0]**2 + data_read[0][1]**2 + data_read[0][2]**2) ** (1/2))
        distance = distance[np.newaxis]
        distance = np.append(distance, np.array((data_read[1][0]**2 + data_read[1][1]**2 + data_read[1][2]**2) ** (1/2)))
        flips = np.zeros(3)
        
        for idx, row in enumerate(data_read):
            if idx >= 2:
                diffs = np.append(diffs, [abs(np.array(data_read[idx]) - np.array(data_read[idx-1]))], axis = 0)
                distance = np.append(distance, (data_read[idx][0]**2 + data_read[idx][1]**2 + data_read[idx][2]**2) ** (1/2))
                old_trend = np.array(data_read[idx-1]) - np.array(data_read[idx-2]) <= 0
                new_trend = np.array(data_read[idx]) - np.array(data_read[idx-1]) >= 0
                flips += old_trend != new_trend
        
        test_features[file][33:36] = np.std(diffs, axis = 0)
        test_features[file][36:39] = np.mean(diffs, axis = 0)
        test_features[file][39:42] = np.percentile(diffs, 25, axis = 0)
        test_features[file][42:45] = np.percentile(diffs, 75, axis = 0)
        test_features[file][45:48] = np.max(diffs, axis = 0)
        test_features[file][48:51] = flips/(0.5*len(data_read))
        test_features[file][51] = np.mean(distance)
        test_features[file][52] = np.std(distance)
        test_features[file][53] = np.min(distance)
        test_features[file][54] = np.max(distance)
        
        counter = 55
        for i in range(nr_rows//windows_size):
            test_features[file][counter:counter+3] = np.mean(data_read[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 3
            test_features[file][counter:counter+3] = np.mean(diffs[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 3
            test_features[file][counter] = np.mean(distance[i*windows_size:windows_size*(i+1)], axis = 0)
            counter += 1
         
        N = len(data_read) # number of sample points
        T = 1/100 # sample spacing
        t = np.linspace(0, N*T, N)
        f = data_read - np.mean(data_read, axis = 0)
        window = hamming(N)
        f = f*window[:, np.newaxis]
        ft = fft(f, axis = 0)
        freq = fftfreq(N,T)[:N//2]
        for row in range(0,3):
            index = {freq[idx]: 0.5*abs(ft[:,row][idx])/N for idx in range(0, len(freq))}
            sorted_values = sorted(list(index.values()), reverse=True)
            test_features[file][counter] = sorted_values[0]
            counter += 1
            test_features[file][counter] = list(index.keys())[list(index.values()).index(sorted_values[0])]
            counter += 1 

test_labels = clf2.predict(np.array(list(test_features.values())))
test_labels = [int(label) for label in test_labels]
with open('submission.csv', 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['id', 'class'])
    for idx in range(0, nr_test):
        writer.writerow([test_files[idx][:-4], test_labels[idx]])
print("submission.csv written successfully.")
        
print(f"Time it took the program to run: {trunc(time() - start_time)} seconds")