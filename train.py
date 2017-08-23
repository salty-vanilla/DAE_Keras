import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import CSVLogger
import argparse
import os

from DataGenerator import DataGenerator
from model import get_model
from callbacks import BatchLogger, ModelSaver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x_dir', type=str)
    parser.add_argument('y_dir', type=str)
    parser.add_argument('--width', '-w', type=int, default=224)
    parser.add_argument('--height', '-ht', type=int, default=224)
    parser.add_argument('--channel', '-ch', type=int, default=3)
    parser.add_argument('--batch_size', '-bs', type=int, default=10)
    parser.add_argument('--nb_epoch', '-e', type=int, default=300)
    parser.add_argument('--nb_sample', '-ns', type=int, default=None)
    parser.add_argument('--param_dir', '-pd', type=str, default="./params")
    parser.add_argument('--color', '-co', type=str, default='rgb')

    args = parser.parse_args()
    x_dir = args.x_dir
    y_dir = args.y_dir

    width = args.width
    height = args.height
    channel = args.channel
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    nb_sample = args.nb_sample
    param_dir = args.param_dir
    color_mode = args.color

    callbacks = [CSVLogger("learning_log_epoch.csv"),
                 BatchLogger("learning_log_iter.csv"),
                 ModelSaver(os.path.join(param_dir, "DAE_{epoch:02d}.hdf5"),
                            save_freq=5)]

    input_shape = (height, width, channel)
    model = get_model(input_shape, is_plot=True)

    model_json_str = model.to_json()
    open(os.path.join(param_dir, "model.json"), 'w').write(model_json_str)

    opt = Adam(lr=1e-3, beta_1=0.1)
    model.compile(opt, 'mse')

    data_gen = DataGenerator(x_dir, y_dir, target_size=input_shape[:2],
                             color_mode=color_mode, nb_sample=nb_sample)

    steps_per_epoch = nb_sample // batch_size if nb_sample % batch_size == 0 else nb_sample // batch_size + 1
    model.fit_generator(data_gen.flow(batch_size), steps_per_epoch=steps_per_epoch,
                        epochs=nb_epoch, callbacks=callbacks)


if __name__ == "__main__":
    main()
