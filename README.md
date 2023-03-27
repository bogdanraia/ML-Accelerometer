# ML-Accelerometer
My most notable solutions to the [PML Smartphone User Identification](https://www.kaggle.com/competitions/pml-2022-smart) competition.
# Task
The task was to classify 3D signals taken over 1,5s from 20 users, sampled at 100Hz. Each user had 450 such examples.
# Models
<div style="text-align: right"> First voting model uses a combination of models - LR, SVM, RF, LGBM - combined with soft probability voting, to obtain best classes on statistical features - min, max, avg, std, fourier features, averaging over time windows, differences.
It obtains 92% accuracy.</div> <br/>
<div style="text-align: right">Second voting model combines a Deep Neural Net on aforementioned features + a deep CNN on the raw signals to get 93% 😁 (can obtain even more with lower learning rate).</div> <br/>
<div style="text-align: right">The RNN is included cause it had potential (it's a simple LSTM over the raw signals) but needs to be combined with a CNN to get good results, probably - it scored at 85% acc.</div>

# Documentation
For a more detailed explanation of my approach, I included the documentation.
