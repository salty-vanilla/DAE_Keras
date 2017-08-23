import numpy as np
import cv2
import os
from keras.preprocessing.image import Iterator
from keras import backend as K
from abc import abstractmethod


class DataGenerator:
    def __init__(self, x_dir, y_dir, target_size=None, color_mode='rgb', nb_sample=None):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.target_size = target_size
        self.color_mode = color_mode
        self.names = os.listdir(x_dir)
        self.x_paths = np.array(sorted([os.path.join(x_dir, f) for f in self.names]))
        self.y_paths = np.array(sorted([os.path.join(y_dir, f) for f in self.names]))
        
        if nb_sample is not None:
            self.x_paths = self.x_paths[:nb_sample]
            self.y_paths = self.y_paths[:nb_sample]

        self.nb_sample = len(self.x_paths)

    @abstractmethod
    def flow(self, batch_size, shuffle=True):
        return DataIterator(x_paths=self.x_paths, y_paths=self.y_paths,
                            target_size=self.target_size, color_mode=self.color_mode,
                            batch_size=batch_size, shuffle=shuffle)


class DataIterator(Iterator):
    def __init__(self, x_paths, y_paths, target_size=None, color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None, is_loop=True):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.target_size = target_size
        self.color_mode = color_mode
        self.is_loop = is_loop
        self.nb_sample = len(self.x_paths)
        super().__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        x_path_batch = self.x_paths[index_array]
        y_path_batch = self.y_paths[index_array]

        x_batch = np.array([load_image(path, self.target_size, self.color_mode)
                            for path in x_path_batch])
        y_batch = np.array([load_image(path, self.target_size, self.color_mode)
                            for path in y_path_batch])
        return x_batch, y_batch


def load_image(path, target_size=None, color_mode='rgb'):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    if color_mode in ['grayscale', 'gray']:
        imread_flag = cv2.IMREAD_GRAYSCALE
        channel = 1
    else:
        imread_flag = cv2.IMREAD_COLOR
        channel = 3

    src = cv2.imread(path, imread_flag)

    if target_size is not None:
        src = cv2.resize(src, target_size, interpolation=cv2.INTER_LINEAR)

    h, w = src.shape[:2]
    if K.image_dim_ordering() == 'th':
        output_shape = (channel, h, w)
    else:
        output_shape = (h, w, channel)
    src = normalize(src)
    return src.reshape(output_shape)


def normalize(x):
    return x.astype('float32') / 255
