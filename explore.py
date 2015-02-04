# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:59:11 2015

@author: mmortonson
"""

import numpy as np
import matplotlib.pyplot as plt


def random_train_images(seed, size):
    from image_files import TrainImages

    n_rows = size[0]
    n_cols = size[1]

    plt.rcParams['figure.figsize'] = (7, 7)

    np.random.seed(seed)

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