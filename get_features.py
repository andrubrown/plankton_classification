# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:07:50 2015

@author: mmortonson
"""

import sys
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_table import FeatureTable


def get_train_features(features, train_file='', scaler_file=''):
    # load training data and compute features if necessary
    if isfile(train_file):
        print 'Loading training data features from {0}'.format(train_file)
        train_data = pd.read_csv(train_file, index_col=0)
        train_classes = train_data['class']
        train_features = train_data[features]
        if isfile(scaler_file):
            scaler_parameters = pd.read_csv(scaler_file)
            scaler_mean = np.array(scaler_parameters['mean'])
            scaler_std = np.array(scaler_parameters['std'])
            scaler = StandardScaler()
            scaler.mean_ = scaler_mean
            scaler.std_ = scaler_std
        else:
            scaler = None
    else:
        print 'Loading training data'
        train = FeatureTable('train')
        print 'Computing features'
        train.add_largest_region_properties()
        scaler = train.scaler

        # save features and scaling parameters
        train.data.to_csv(train_file)
        if not isfile(scaler_file):
            scaler_df = pd.DataFrame({'mean': scaler.mean_,
                                      'std': scaler.std_})
            scaler_df.to_csv(scaler_file)

        # split into class labels and features
        train_classes = train.data['class']
        train_features = train.data[features]

    if scaler is None:
        return (train_features, train_classes)
    else:
        return (train_features, train_classes, scaler)


def get_test_features(features, scaler=None, test_file=''):
    # load test data and compute features, using scaling from training data
    if isfile(test_file):
        print '\nLoading test data features from {0}'.format(test_file)
        test_data = pd.read_csv(test_file, index_col=0)
        test_features = test_data[features]
        test_index = test_data.index
    else:
        if scaler is None:
            sys.exit('Missing scaler used on training data features.')
        print '\nLoading test data'
        test = FeatureTable('test')
        print 'Computing features'
        test.add_largest_region_properties(scaler=scaler)
        test_features = test.data[features]
        test_index = test.data.index

        # save features
        test.data.to_csv(test_file)

    return (test_features, test_index)