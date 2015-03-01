# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:02 2015

@author: mmortonson
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from feature_table import FeatureTable
from sklearn.grid_search import GridSearchCV

features = ['solidity', 'max_intensity', 'min_intensity',
            'filled_area', 'euler_number', 'eccentricity']


# load training data and compute features
print 'Loading training data'
train = FeatureTable('train')
print 'Computing features'
train.add_largest_region_properties()

# save features and scaling parameters
train.data.to_csv('train_features.csv')
print 'Feature scaling:'
print 'mean values:'
print train.scaler.mean_
print 'standard deviations:'
print train.scaler.std_

# split data into class labels and features
train_classes = train.data['class']
train_features = train.data.drop(['class'], axis=1)

# optimize k-NN classifier by finding k with best log-loss on CV samples
print 'Optimizing k-NN'
knn = KNeighborsClassifier()
knn_opt = GridSearchCV(knn, {'n_neighbors': [800, 850, 900, 950]},
                       scoring='log_loss')
knn_opt.fit(train_features, train_classes)
print knn_opt.grid_scores_

# load test data and compute features, using scaling from training data
print 'Loading test data'
test = FeatureTable('test')
print 'Computing features'
test.add_largest_region_properties(features, train.scaler)
test_features = test.data.drop(['class'], axis=1)

# save features
test.data.to_csv('test_features.csv')

# predict class probabilities for test data
p_class = knn_opt.predict_proba(test_features)

# write probabilities to output file
classes = sorted(np.unique(train_classes))
files = [str(i) + '.jpg' for i in test.data.index]
p_class_df = pd.DataFrame(p_class, index=files, columns=classes)
p_class_df.to_csv('predictions/test_prob_knn.csv', index_label='image')
