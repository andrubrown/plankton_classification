# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:20:11 2015

@author: mmortonson
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from image_files import TrainImages, TestImages


class FeatureTable(object):
    def __init__(self, sample, max_rows=None):
        if sample == 'train':
            self.image_set = TrainImages()
        elif sample == 'test':
            self.image_set = TestImages()
        else:
            raise ValueError
        ids = self.image_set.get_all_ids()
        if max_rows is not None:
            ids = ids[:max_rows]
        classes = [self.image_set.get_class_by_id(int(i)) for i in ids]
        self.data = pd.DataFrame(data={'class': classes}, index=ids)
        self.scaler = None

    def add_largest_region_properties(self, properties, scaler=None):
        table = {}
        for i in self.data.index:
            image = self.image_set.get_image_by_id(int(i))
            region_props = image.get_largest_region_properties()
            for p in properties:
                value = getattr(region_props, p)
                try:
                    for j, v in enumerate(value):
                        key = p + str(j)
                        table[key] = table.get(key, []) + [v]
                except:
                    table[p] = table.get(p, []) + [value]
        new_data = pd.DataFrame(data=table, index=self.data.index)
        indices = new_data.index
        columns = new_data.columns
        if scaler is None:
            self.scaler = StandardScaler()
            new_data = self.scaler.fit_transform(new_data)
        else:
            self.scaler = scaler
            new_data = self.scaler.transform(new_data)
        new_data = pd.DataFrame(new_data, index=indices, columns=columns)
        self.data = pd.concat([self.data, new_data], axis=1)

    def pair_plot_random_classes(self, n=2, properties=None, seed=None,
                                 n_example_images=0):
        if seed is not None:
            np.random.seed(seed)
        all_classes = self.data['class'].unique()
        classes = np.random.choice(all_classes, n, replace=False)
        rows = self.data['class'].isin(classes)
        if properties is None:
            plot_data = self.data[rows]
        else:
            plot_data = self.data[rows][['class'] + properties]

        plt.figure(figsize=(7, 9))
        sns.set_style('darkgrid')
        grid = sns.PairGrid(plot_data,
                            hue='class', palette='Set2',
                            size=9.0/len(plot_data.columns))
        grid.map_offdiag(plt.scatter, alpha=0.7)
        grid.map_diag(sns.kdeplot)
        grid.add_legend()

        if n_example_images > 0:
            sns.set_style('white')
            for i, cl in enumerate(classes):
                plt.figure(figsize=(7, 1))
                plt.suptitle(cl, fontsize=15, y=1.2)
                images = self.image_set.get_random_images_in_class(
                    cl, n_example_images)
                for j, im in enumerate(images):
                    plt.subplot(1, n_example_images, j+1)
                    im.plot_largest_region()
                    sns.despine(left=True, bottom=True)