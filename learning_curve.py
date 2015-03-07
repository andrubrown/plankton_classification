# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:26:25 2015

@author: mmortonson
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from get_features import get_train_features


clf = SVC(kernel='rbf', probability=True, class_weight=None,
          C=100, gamma=0.1)

features = ['solidity', 'max_intensity', 'min_intensity',
            'filled_area', 'euler_number', 'eccentricity']
            #'weighted_moments_hu0', 'weighted_moments_hu1',
            #'weighted_moments_hu2', 'weighted_moments_hu3']

cv_percent = 30


train_features, train_classes = \
    get_train_features(features, train_file='train_features.csv',
                       scaler_file='feature_scaling.csv')

n_train_total = len(train_classes)
n_train_max = n_train_total * (100 - cv_percent)/100
train_cv_features = train_features.iloc[:n_train_max]
train_cv_classes = train_classes.iloc[:n_train_max]
test_cv_features = train_features.iloc[n_train_max:]
test_cv_classes = train_classes.iloc[n_train_max:]

n_train = []
train_error = []
test_error = []

for percent in range(10, 110, 10):
    print 'Training with {0}% of data'.format(percent)
    n = n_train_max * percent/100
    n_train.append(n)
    train_sub_classes = pd.Series([])
    while len(train_sub_classes.unique()) < len(train_classes.unique()):
        indices = np.random.choice(train_cv_classes.index, n, replace=False)
        train_sub_classes = train_cv_classes[indices]
    train_sub_features = train_cv_features.loc[indices]
    clf.fit(train_sub_features, train_sub_classes)
    train_error.append(log_loss(train_sub_classes,
                                clf.predict_proba(train_sub_features)))
    test_error.append(log_loss(test_cv_classes,
                               clf.predict_proba(test_cv_features)))

plt.plot(n_train, train_error, 'b', label='Training error')
plt.plot(n_train, test_error, 'r', label='Test error')
plt.xlabel('Number of samples')
plt.ylabel('log loss')
plt.legend()
plt.show()
