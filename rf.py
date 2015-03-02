# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:02 2015

@author: mmortonson
"""

import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from get_features import get_train_features, get_test_features


features = ['solidity', 'max_intensity', 'min_intensity',
            'filled_area', 'euler_number', 'eccentricity']

prediction_file = 'predictions/test_prob_rf.csv'


train_features, train_classes = \
    get_train_features(features, train_file='train_features.csv',
                       scaler_file='feature_scaling.csv')

# use CV to find classifier settings that optimize log loss
clf = RandomForestClassifier()
n_estimators = []
while len(n_estimators) != 1:
    input_string = raw_input('\nEnter multiple values of ' +
                             'the number of estimators' +
                             ' to try with CV,' +
                             '\nor one value to train the classifier:\n')
    n_estimators = [int(x) for x in input_string.split()]
    clf_opt = GridSearchCV(clf, {'n_estimators': n_estimators},
                           scoring='log_loss', cv=5)
    clf_opt.fit(train_features, train_classes)
    if len(n_estimators) > 1:
        print 'log loss estimates:'
        for cv_score in clf_opt.grid_scores_:
            print cv_score

test_features, test_index = get_test_features(features,
                                              test_file='test_features.csv')

print 'Predicting class probabilities for test data'
p_class = clf_opt.predict_proba(test_features)
classes = sorted(np.unique(train_classes))
files = [str(i) + '.jpg' for i in test_index]
p_class_df = pd.DataFrame(p_class, index=files, columns=classes)
p_class_df.to_csv(prediction_file, index_label='image')
print 'Probabilities written to {0}'.format(prediction_file)
