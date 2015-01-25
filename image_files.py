# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 10:31:38 2015

@author: mmortonson
"""

"""
Methods:
- get a specific image by file number
- get one or more random images:
  - from the training set
  - from the training set with a single random class
  - from the training set with a specified class
  - from the test set

Compute features for images and plot a set of images, labeled by file name,
class (if known), and feature values.
Use random seeds for repeatability.
"""


import os
import glob
import pandas as pd
import skimage.io


class ImageSet(object):
    def __init__(self):
        self.data = pd.DataFrame()

    def get_image_number(self, number):
        if type(number) == int:
            return self.data.loc[number, 'image']
        else:
            raise ValueError


class TestImages(ImageSet):
    def __init__(self):
        filepaths = glob.glob(os.path.join('competition_data',
                                           'test', '*.jpg'))
        filenumbers = [int(p.split(os.sep)[-1].split('.')[0])
                       for p in filepaths]
        images = [ImageFile(p) for p in filepaths]
        classes = ['unknown' for p in filepaths]
        d = {'image': images, 'class': classes}
        self.data = pd.DataFrame(data=d, index=filenumbers)
        self.data.sort_index(inplace=True)


class TrainImages(ImageSet):
    def __init__(self):
        pass


class TrainClasses(object):
    def __init__(self):
        rel_paths = sorted(list(set(glob.glob(
            os.path.join('competition_data', 'train', '*'))).difference(
            set(glob.glob(os.path.join('competition_data', 'train', '*.*'))))))
        self.names = []
        self.path = {}
        self.filepaths = {}
        self.files = {}
        for p in rel_paths:
            name = p.split(os.sep)[-1]
            self.names.append(name)
            self.path[name] = os.path.abspath(p)
            self.filepaths[name] = glob.glob(os.path.join(self.path[name],
                                                          '*.jpg'))
            self.files[name] = [f.split(os.sep)[-1] for f in self.filepaths]

    def get_class_names(self):
        return self.names

    def get_path(self, class_name):
        return self.path.get(class_name, '')

    def get_image_files(self, class_name):
        return self.files.get(class_name, [])

    def get_image_filepaths(self, class_name):
        return self.filepaths.get(class_name, [])

    def n_images(self, class_name):
        return len(self.get_image_files(class_name))


class ImageFile(object):
    def __init__(self, filepath):
        if filepath[-4:] != '.jpg' or not os.path.exists(filepath):
            raise ValueError
        self.filepath = filepath
        self.image = None

    def get_image(self):
        if self.image is None:
            self.image = skimage.io.imread(self.filepath, as_grey=True)
        return self.image

    def plot(self):
        skimage.io.imshow(self.get_image())
