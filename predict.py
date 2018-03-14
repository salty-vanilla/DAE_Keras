import os
import sys
import numpy as np
from PIL import Image
import argparse
import keras
from image_sampler import ImageSampler, denormalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('param_path', type=str)
    parser.add_argument('image_dir', type=str)
    parser.add_argument('--result_dir', '-rd', type=str, default='./result')
    parser.add_argument('--width', '-w', type=int, default=224)
    parser.add_argument('--height', '-ht', type=int, default=224)
    parser.add_argument('--channel', '-ch', type=int, default=3)
    parser.add_argument('--batch_size', '-bs', type=int, default=10)
    parser.add_argument('--color', '-co', type=str, default='rgb')

    args = parser.parse_args()

    autoencoder = keras.models.model_from_json(open(args.model_path).read())
    autoencoder.load_weights(args.param_path)

    data_gen = ImageSampler(args.image_dir,
                            None,
                            target_size=(args.width, args.height),
                            color_mode=args.color,
                            ).flow(batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = data_gen.n // data_gen.batch_size if data_gen.n % data_gen.batch_size == 0 \
        else data_gen.n // data_gen.batch_size + 1

    os.makedirs(os.path.join(args.result_dir, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'concat'), exist_ok=True)

    for step in range(steps_per_epoch):
        image_batch = next(data_gen)
        pred = autoencoder.predict_on_batch(image_batch)

        for i, (im, p) in enumerate(zip(image_batch, pred)):
            im = denormalize(im)
            p = denormalize(p)
            c = np.concatenate([im, p], axis=1)

            c = Image.fromarray(c)
            p = Image.fromarray(p)

            c.save(os.path.join(args.result_dir, 'concat', '{}_{}.png'.format(step, i)))
            p.save(os.path.join(args.result_dir, 'pred', '{}_{}.png'.format(step, i)))


if __name__ == '__main__':
    main()
