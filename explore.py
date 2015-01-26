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
        image.plot()
        plt.title(train.get_class_by_id(image.get_id()),
                  fontdict={'fontsize': 9})

    plt.tight_layout(1.08, h_pad=1, w_pad=0)
    plt.show()