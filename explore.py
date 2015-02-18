# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:59:11 2015

@author: mmortonson
"""

import numpy as np
import matplotlib.pyplot as plt
from image_files import TrainClasses, TrainImages


def feature_comparison_random_class_pair(features, seed=None):
    if seed is not None:
        np.random.seed(seed)
    classes = np.random.choice(TrainClasses().get_class_names(), size=2,
                               replace=False)
    if len(features) == 1:
        feature_histogram_comparison(features[0], classes)
    else:
        feature_scatter_comparison(features, classes)


def feature_histogram_comparison(feature, classes):
    all_feature_values = []
    all_classes = TrainClasses()
    for c in classes:
        if c in all_classes.get_class_names():
            class_feature_values = compute_class_features(c, feature)
            all_feature_values.append(class_feature_values)

    plt.hist(all_feature_values, histtype='step', label=classes)
    plt.xlabel(feature)
    plt.legend()
    plt.show()


def feature_scatter_comparison(features, classes):
    all_classes = TrainClasses()
    colors = ['k', 'r', 'b', 'g']
    for c, color in zip(classes, colors):
        if c in all_classes.get_class_names():
            feature_values = {}
            for f in features:
                feature_values[f] = compute_class_features(c, f)
            plt.scatter(feature_values[features[0]], feature_values[features[1]],
                        color=color, alpha=0.5, label=c)

    plt.legend()
    plt.show()


def compute_class_features(class_name, feature):
    all_classes = TrainClasses()
    feature_values = []
    for image in all_classes.get_image_files(class_name):
        props = image.get_largest_region_properties()
        feature_values.append(getattr(props, feature))
    return feature_values


def random_train_images(size, seed=None):

    if seed is not None:
        np.random.seed(seed)

    n_rows = size[0]
    n_cols = size[1]

    plt.rcParams['figure.figsize'] = (7, 7)

    train = TrainImages()

    for i, image in enumerate(train.get_random_images(n_rows*n_cols)):
        plt.subplot(n_rows, n_cols, i+1)
        image.plot_largest_region()
        props = image.get_largest_region_properties()
        format_string = '{0}\nFilled area: {1}\nEccentricity: {2:.2f}' + \
            '\nEuler number: {3}\nIntensity range: {4}, {5}' + \
            '\nSolidity: {6:.2f}\nWeighted Hu moments:\n' + \
            '{7:.2e} {8:.2e} {9:.2e} {10:.2e}\n{11:.2e} {12:.2e} {13:.2e}'
        plt.title(format_string.format(
            train.get_class_by_id(image.get_id()),
            props.filled_area,
            props.eccentricity,
            props.euler_number,
            props.min_intensity,
            props.max_intensity,
            props.solidity,
            *props.weighted_moments_hu),
            fontdict={'fontsize': 9})

    plt.tight_layout(1.08, h_pad=3, w_pad=0)
    plt.show()