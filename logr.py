# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:02 2015

@author: mmortonson
"""

import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from get_features import get_train_features, get_test_features


features = ['solidity', 'max_intensity', 'min_intensity',
            'filled_area', 'euler_number', 'eccentricity',
            'weighted_moments_hu0', 'weighted_moments_hu1',
            'weighted_moments_hu2', 'weighted_moments_hu3']

prediction_file = 'predictions/test_prob_logr_whm.csv'


train_features, train_classes = \
    get_train_features(features, train_file='train_features.csv',
                       scaler_file='feature_scaling.csv')

# use CV to find classifier settings that optimize log loss
clf = LogisticRegression()
clf_trained = False
C_values = [0]
while len(C_values) > 0:
    input_string = raw_input('\nEnter one or more values of C' +
                             ' to try with CV,' +
                             '\nor press Enter to predict classes using ' +
                             'the best result from the previous grid:\n')
    C_values = [float(x) for x in input_string.split()]
    if len(C_values) > 0:
        clf_opt = GridSearchCV(clf, {'C': C_values},
                               scoring='log_loss', cv=5)
        clf_opt.fit(train_features, train_classes)
        clf_trained = True
        print 'Estimated log loss:'
        for cv_score in clf_opt.grid_scores_:
            print cv_score
    elif not clf_trained:
        print 'The classifier has not been trained yet.'
        C_values = [0]

test_features, test_index = get_test_features(features,
                                              test_file='test_features.csv')

print 'Predicting class probabilities for test data'
p_class = clf_opt.predict_proba(test_features)
classes = sorted(np.unique(train_classes))
files = [str(i) + '.jpg' for i in test_index]
p_class_df = pd.DataFrame(p_class, index=files, columns=classes)
p_class_df.to_csv(prediction_file, index_label='image')
print 'Probabilities written to {0}'.format(prediction_file)
