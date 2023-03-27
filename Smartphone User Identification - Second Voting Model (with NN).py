# Smartphone User Identification
# Voting model using a NN and a CNN

import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from math import trunc
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

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

train_data = np.empty((nr_train, nr_rows, 3))
valid_data = np.empty((nr_validation, nr_rows, 3))
test_data = np.empty((nr_test, nr_rows, 3))

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

def create_model_cnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(16, 5, padding="same", activation='relu', input_shape=(139, 3,)))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(32, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(64, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(128, 2, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(256, 2, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(512, 2, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))  
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(20, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00009), metrics=['sparse_categorical_accuracy'])
    return model

def create_model_nn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(120, input_shape=(nr_features + nr_windows_features*nr_rows//windows_size,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), metrics=['sparse_categorical_accuracy'])
    return model

# Read the labels from "train_labels.csv", put them in train_labels
with open("train_labels.csv") as fp:
    label_file = csv.reader(fp)
    data_read = [row for row in label_file]

for idx, row in enumerate(data_read[1:]):
    if idx < nr_train:
        train_labels[row[0]] = row[1]
    else:
        valid_labels[row[0]] = row[1]

for idx2, file in enumerate(train_files):
    file_name = os.path.join(train_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [[float(x) for x in row] for row in reader]
        train_data[idx2][0:nr_rows] = np.array(data_read[0:nr_rows])
        file = file[:-4]
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

for idx2, file in enumerate(valid_files):
    file_name = os.path.join(train_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [[float(x) for x in row] for row in reader]
        # data_read = [float(x) for row in reader for x in row] 
        valid_data[idx2][0:nr_rows] = np.array(data_read[0:nr_rows])
        file = file[:-4]
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

cnn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_cnn, epochs=150, batch_size = 64, 
                                                           callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                                                           validation_data=(np.array(valid_data), np.array([int(x) - 1 for x in list(valid_labels.values())])))
cnn_model.fit(np.array(train_data), np.array([int(x) - 1 for x in list(train_labels.values())]))
predicted_labels = cnn_model.predict(np.array(valid_data))
# cnn_cv_scores = cross_val_score(cnn_model, np.concatenate((train_data, valid_data)), np.array([int(x) - 1 for x in list(train_labels.values())] + [int(x) - 1 for x in list(valid_labels.values())]))
# print(f"5-cv accuracy for CNN model: {np.mean(cnn_cv_scores)}")
print(f"Validation accuracy for CNN model: {accuracy_score(list(valid_labels.values()), [str(label + 1) for label in predicted_labels])}")
    
nn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_nn, epochs=150, batch_size = 64, 
                                                           callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                                                           validation_data=(np.array(list(valid_features.values())), np.array([int(x) - 1 for x in list(valid_labels.values())])))
nn_model.fit(np.array(list(train_features.values())), np.array([int(x) - 1 for x in list(train_labels.values())]))
predicted_labels = nn_model.predict(np.array(list(valid_features.values())))
# nn_cv_scores = cross_val_score(nn_model, np.array(list(train_features.values()) + list(valid_features.values())), np.array([int(x) - 1 for x in list(train_labels.values())] + [int(x) - 1 for x in list(valid_labels.values())]))
# print(f"5-cv accuracy for NN model: {np.mean(nn_cv_scores)}")
print(f"Validation accuracy for NN model: {accuracy_score(list(valid_labels.values()), [str(label + 1) for label in predicted_labels])}")

cnn_score = cnn_model.predict_proba(np.array(valid_data), batch_size = 64)
nn_score = nn_model.predict_proba(np.array(list(valid_features.values())), batch_size = 64)

predicted_labels = cnn_score + nn_score
predicted_labels = [str(np.argmax(label) + 1) for label in predicted_labels]
print(f"Validation accuracy for voting classifier: {accuracy_score(list(valid_labels.values()), predicted_labels)}")
cm = confusion_matrix(list(valid_labels.values()), predicted_labels)
cmp = ConfusionMatrixDisplay(cm, display_labels = np.arange(1, 21))
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax)

cnn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_cnn, epochs=150, batch_size = 64, 
                                                           callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10))
cnn_model.fit(np.concatenate((train_data, valid_data)), np.array([int(x) - 1 for x in list(train_labels.values())] + [int(x) - 1 for x in list(valid_labels.values())]))

nn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_nn, epochs=150, batch_size = 64, 
                                                           callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10))
nn_model.fit(np.array(list(train_features.values()) + list(valid_features.values())), np.array([int(x) - 1 for x in list(train_labels.values())] + [int(x) - 1 for x in list(valid_labels.values())]))

for idx2, file in enumerate(test_files):
    file_name = os.path.join(test_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [[float(x) for x in row] for row in reader]
        # data_read = [float(x) for row in reader for x in row] 
        test_data[idx2][0:nr_rows] = np.array(data_read[0:nr_rows])
        file = file[:-4]
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

cnn_score = cnn_model.predict_proba(np.array(test_data), batch_size = 64)
nn_score = nn_model.predict_proba(np.array(list(test_features.values())), batch_size = 64)

test_labels = cnn_score + nn_score
test_labels = [str(np.argmax(label) + 1) for label in test_labels]
with open('submission.csv', 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['id', 'class'])
    for idx in range(0, nr_test):
        writer.writerow([test_files[idx][:-4], test_labels[idx]])
print("submission.csv written successfully.")
        
print(f"Time it took the program to run: {trunc(time() - start_time)} seconds")