import os
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
import argparse

from image_sampler import ImageSampler
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
    parser.add_argument('--save_step', '-ss', type=int, default=5)
    parser.add_argument('--nb_sample', '-ns', type=int, default=None)
    parser.add_argument('--param_dir', '-pd', type=str, default="./params")
    parser.add_argument('--color', '-co', type=str, default='rgb')

    args = parser.parse_args()

    callbacks = [CSVLogger("learning_log_epoch.csv"),
                 BatchLogger("learning_log_iter.csv"),
                 ModelSaver(os.path.join(args.param_dir, "DAE_{epoch:03d}.hdf5"),
                            save_freq=args.save_step)]

    input_shape = (args.height, args.width, args.channel)
    model = get_model(input_shape, is_plot=False)

    model_json_str = model.to_json()
    open(os.path.join(args.param_dir, "model.json"), 'w').write(model_json_str)

    opt = Adam(lr=1e-3, beta_1=0.1)
    model.compile(opt, 'mse')

    data_gen = ImageSampler(args.x_dir,
                            args.y_dir,
                            target_size=(args.width, args.height),
                            color_mode=args.color,
                            nb_sample=args.nb_sample)

    steps_per_epoch = args.nb_sample // args.batch_size if args.nb_sample % args.batch_size == 0 \
        else args.nb_sample // args.batch_size + 1
    model.fit_generator(data_gen.flow(args.batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.nb_epoch,
                        callbacks=callbacks)


if __name__ == "__main__":
    main()
