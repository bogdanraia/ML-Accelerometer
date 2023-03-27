# Smartphone User Identification
# RNN

import os
import csv
import numpy as np
import tensorflow as tf
from time import time
from math import trunc
from sklearn.metrics import accuracy_score

# TODO: Find a better alternative than feature position hardcoding
np.set_printoptions(suppress=True) # Remove scientific notation

start_time = time()
nr_classes = 20
nr_validation = 2000
nr_test = 5000
nr_train = 7000

windows_size = 26
nr_rows = 130

train_data = np.empty((nr_train, nr_rows, 3))
valid_data = np.empty((nr_validation, nr_rows, 3))
test_data = np.empty((nr_test, nr_rows, 3))

train_labels = {}
train_folder = 'train' # copied train/test folders into main directory, without the second layer of train/test folders
train_files = os.listdir(train_folder)   

valid_labels = {}

test_labels = {}
test_folder = 'test'
test_files = os.listdir(test_folder)

valid_files = train_files[-nr_validation:]
train_files = train_files[:-nr_validation] # ignore validation files

def create_model_rnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape= [130, 3]))
    model.add(tf.keras.layers.Dense(nr_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), metrics=['sparse_categorical_accuracy'])
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

for idx2, file in enumerate(valid_files):
    file_name = os.path.join(train_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [[float(x) for x in row] for row in reader]
        valid_data[idx2][0:nr_rows] = np.array(data_read[0:nr_rows])
        
rnn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_rnn, epochs=150, batch_size = 256, 
                                                           callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                                                           validation_data=(np.array(valid_data), np.array([int(x) - 1 for x in list(valid_labels.values())])))
rnn_model.fit(np.array(train_data), np.array([int(x) - 1 for x in list(train_labels.values())]))
predicted_labels = rnn_model.predict(np.array(valid_data))
# cnn_cv_scores = cross_val_score(rnn_model, np.concatenate((train_data, valid_data)), np.array([int(x) - 1 for x in list(train_labels.values())] + [int(x) - 1 for x in list(valid_labels.values())]))
# print(f"5-cv accuracy for RNN model: {np.mean(cnn_cv_scores)}")
print(f"Validation accuracy for RNN model: {accuracy_score(list(valid_labels.values()), [str(label + 1) for label in predicted_labels])}")
  
for idx2, file in enumerate(test_files):
    file_name = os.path.join(test_folder, file)
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [[float(x) for x in row] for row in reader]
        # data_read = [float(x) for row in reader for x in row] 
        test_data[idx2][0:nr_rows] = np.array(data_read[0:nr_rows])
        file = file[:-4]

rnn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_rnn, epochs=50, batch_size = 64)
rnn_model.fit(np.concatenate((train_data, valid_data)), np.array([int(x) - 1 for x in list(train_labels.values())] + [int(x) - 1 for x in list(valid_labels.values())]))
rnn_score = rnn_model.predict(np.array(test_data), batch_size = 64) + 1

with open('submission.csv', 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['id', 'class'])
    for idx in range(0, nr_test):
        writer.writerow([test_files[idx][:-4], rnn_score[idx]])
print("submission.csv written successfully.")
        
print(f"Time it took the program to run: {trunc(time() - start_time)} seconds")