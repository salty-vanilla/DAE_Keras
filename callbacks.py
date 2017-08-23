# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import keras.callbacks
try:
    import cPickle as pickle
except:
    import pickle
import csv
from collections import deque
from collections import OrderedDict
from collections import Iterable


class BatchLogger(keras.callbacks.CSVLogger):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.on_epoch_end = keras.callbacks.Callback.on_epoch_end

        dst_dir = os.path.dirname(file_path)
        if dst_dir is not '':
            os.makedirs(dst_dir, exist_ok=True)

    def on_batch_end(self, batch, logs=None):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['batch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class ModelSaver(keras.callbacks.ModelCheckpoint):
    def __init__(self, file_path, verbose=0, save_freq=1):
        super().__init__(file_path, verbose=verbose)
        self.save_freq = save_freq

        dst_dir = os.path.dirname(file_path)
        if dst_dir is not '':
            os.makedirs(dst_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            super().on_epoch_end(epoch, logs=logs)


# TODO segmentation用に作る
class Visualizer(keras.callbacks.Callback):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def on_epoch_end(self, epoch, logs=None):
        predict = self.model.predict(self.x)
        print(predict.shape)
