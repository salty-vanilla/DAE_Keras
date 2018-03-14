import numpy as np
from PIL import Image
import os
from keras.preprocessing.image import Iterator


class ImageSampler:
    def __init__(self, x_dir,
                 y_dir,
                 target_size=None,
                 color_mode='rgb',
                 nb_sample=None):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.target_size = target_size
        self.color_mode = color_mode
        self.names = os.listdir(x_dir)
        self.x_paths = np.array(sorted([os.path.join(x_dir, f) for f in self.names]))

        if y_dir is None:
            self.y_paths = None
        else:
            self.y_paths = np.array(sorted([os.path.join(y_dir, f) for f in self.names]))

        if nb_sample is not None:
            self.x_paths = self.x_paths[:nb_sample]
            self.y_paths = self.y_paths[:nb_sample]

        self.nb_sample = len(self.x_paths)

    def flow(self, batch_size, shuffle=True):
        return DataIterator(x_paths=self.x_paths, y_paths=self.y_paths,
                            target_size=self.target_size, color_mode=self.color_mode,
                            batch_size=batch_size, shuffle=shuffle)


class DataIterator(Iterator):
    def __init__(self, x_paths,
                 y_paths,
                 target_size=None,
                 color_mode='rgb',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 is_loop=True):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.target_size = target_size
        self.color_mode = color_mode
        self.is_loop = is_loop
        self.nb_sample = len(self.x_paths)
        super().__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, _, _ = next(self.index_generator)
        x_path_batch = self.x_paths[index_array]
        x_batch = np.array([load_image(path, self.target_size, self.color_mode)
                            for path in x_path_batch])

        if self.y_paths is not None:
            y_path_batch = self.y_paths[index_array]
            y_batch = np.array([load_image(path, self.target_size, self.color_mode)
                                for path in y_path_batch])
            return x_batch, y_batch
        else:
            return x_batch


def load_image(path, target_size=None, color_mode='rgb'):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    image = Image.open(path)

    if color_mode in ['grayscale', 'gray']:
        image = image.convert('L')

    if target_size is not None and target_size != image.size:
        image = image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(image)
    image_array = normalize(image_array)

    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    return image_array


def normalize(x):
    return x.astype('float32') / 255


def denormalize(x):
    return (x * 255).astype('uint8')
