# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:53:20 2015

@author: mmortonson
"""

import numpy as np
from feature_table import FeatureTable


# load training data and compute features
t = FeatureTable('train')
t.add_largest_region_properties()

# group feature data by class label
grouped_classes = t.data.groupby('class')

# compute mean and variance of features within each class
mean_var = grouped_classes.aggregate([np.mean, np.var])

# compute variance of means and mean of variances,
# then find the ratio
grouped_features = mean_var.stack(level=0).groupby(level=1)
feature_variation = grouped_features.aggregate({'mean': np.var,
                                                'var': np.mean})
ratio = feature_variation['mean'] / feature_variation['var']
print ratio
