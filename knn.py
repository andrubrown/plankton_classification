# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:02 2015

@author: mmortonson
"""

import sys
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from feature_table import FeatureTable

features = ['solidity', 'max_intensity', 'min_intensity',
            'filled_area', 'euler_number', 'eccentricity']

prediction_file = 'predictions/test_prob_knn_weighted.csv'

train_file = 'train_features.csv'
scaler_file = 'feature_scaling.csv'
test_file = 'test_features.csv'


# load training data and compute features if necessary
if isfile(train_file):
    print 'Loading training data features from {0}'.format(train_file)
    train_data = pd.read_csv(train_file, index_col=0)
    train_classes = train_data['class']
    train_features = train_data[features]
    if not isfile(test_file):
        if isfile(scaler_file):
            scaler_parameters = pd.read_csv(scaler_file)
            scaler_mean = np.array(scaler_parameters['mean'])
            scaler_std = np.array(scaler_parameters['std'])
            scaler = StandardScaler()
            scaler.mean_ = scaler_mean
            scaler.std_ = scaler_std
        else:
            sys.exit('Missing scaling parameters from training data.')
else:
    print 'Loading training data'
    train = FeatureTable('train')
    print 'Computing features'
    train.add_largest_region_properties()
    scaler = train.scaler

    # save features and scaling parameters
    train.data.to_csv(train_file)
    if not isfile(scaler_file):
        scaler_df = pd.DataFrame({'mean': scaler.mean_, 'std': scaler.std_})
        scaler_df.to_csv(scaler_file)

    # split into class labels and features
    train_classes = train.data['class']
    train_features = train.data[features]


# optimize k-NN classifier by finding k with best log-loss on CV samples
knn = KNeighborsClassifier(weights='distance')
n_neighbors = []
while len(n_neighbors) != 1:
    input_string = raw_input('\nEnter multiple values of k to try with CV,' +
                             '\nor one value to train the k-NN classifier:\n')
    n_neighbors = [int(x) for x in input_string.split()]
    knn_opt = GridSearchCV(knn, {'n_neighbors': n_neighbors},
                           scoring='log_loss', cv=5, verbose=1)
    knn_opt.fit(train_features, train_classes)
    if len(n_neighbors) > 1:
        print 'log loss estimates:'
        for cv_score in knn_opt.grid_scores_:
            print cv_score

# load test data and compute features, using scaling from training data
if isfile(test_file):
    print '\nLoading test data features from {0}'.format(test_file)
    test_data = pd.read_csv(test_file, index_col=0)
    test_features = test_data[features]
    test_index = test_data.index
else:
    print '\nLoading test data'
    test = FeatureTable('test')
    print 'Computing features'
    test.add_largest_region_properties(scaler=scaler)
    test_features = test.data[features]
    test_index = test.data.index

    # save features
    test.data.to_csv(test_file)

print 'Predicting class probabilities for test data'
p_class = knn_opt.predict_proba(test_features)
classes = sorted(np.unique(train_classes))
files = [str(i) + '.jpg' for i in test_index]
p_class_df = pd.DataFrame(p_class, index=files, columns=classes)
p_class_df.to_csv(prediction_file, index_label='image')
print 'Probabilities written to {0}'.format(prediction_file)
