# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:20:11 2015

@author: mmortonson
"""

import pandas as pd
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

    def add_largest_region_properties(self, properties):
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
        self.data = pd.concat([self.data,
                               pd.DataFrame(data=table,
                                            index=self.data.index)],
                              axis=1)
