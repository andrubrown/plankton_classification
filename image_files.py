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
import numpy as np
import pandas as pd
import skimage.io
from skimage import morphology
import matplotlib.pyplot as plt
from pylab import cm


class ImageSet(object):
    def __init__(self):
        self.data = pd.DataFrame()

    def get_image_by_id(self, id):
        if type(id) == int:
            return self.data.loc[id, 'image']
        else:
            raise ValueError

    def get_class_by_id(self, id):
        if type(id) == int:
            return self.data.loc[id, 'class']
        else:
            raise ValueError

    def get_random_images(self, how_many=1):
        image_numbers = np.random.choice(self.data.index, size=how_many,
                                         replace=False)
        return list(self.data['image'][image_numbers])


class TestImages(ImageSet):
    def __init__(self):
        filepaths = glob.glob(os.path.join('competition_data',
                                           'test', '*.jpg'))
        filenumbers = [int(p.split(os.sep)[-1][:-4]) for p in filepaths]
        images = [ImageFile(p) for p in filepaths]
        classes = ['unknown' for p in filepaths]
        d = {'image': images, 'class': classes}
        self.data = pd.DataFrame(data=d, index=filenumbers)
        self.data.sort_index(inplace=True)


class TrainImages(ImageSet):
    def __init__(self):
        train_classes = TrainClasses()
        filenumbers = []
        images = []
        classes = []
        for c in train_classes.get_class_names():
            ids = [int(p.split(os.sep)[-1][:-4]) for p in
                   train_classes.get_image_filepaths(c)]
            filenumbers.extend(ids)
            images.extend(train_classes.get_image_files(c))
            classes.extend(list(np.repeat(c, train_classes.n_images(c))))
        d = {'image': images, 'class': classes}
        self.data = pd.DataFrame(data=d, index=filenumbers)
        self.data.sort_index(inplace=True)


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
            self.files[name] = [ImageFile(f) for f in self.filepaths[name]]

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
        self.id = int(filepath.split(os.sep)[-1][:-4])
        self.image = None

    def get_image(self):
        if self.image is None:
            self.image = skimage.io.imread(self.filepath, as_grey=True)
        return self.image

    def get_id(self):
        return self.id

    def plot(self):
        plt.imshow(self.get_image(), cmap=cm.gray_r)

    def restore_original(self):
        self.image = None
        self.get_image()

    def threshold(self, level='mean'):
        im = self.get_image()
        if level == 'mean':
            level = np.mean(im)
        self.image = np.where(im > level, 0., 1.)

    def dilate(self, size=4):
        im = self.get_image()
        self.image = morphology.dilation(im, np.ones((4, 4)))
