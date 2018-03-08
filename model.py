from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K


def _down_block(x,
                filters,
                conv_iteration=1,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same'):

    for _ in range(conv_iteration):
        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   activation=None
                   )(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    return x


def _up_block(x,
              filters,
              channel_reduction=True,
              conv_iteration=1,
              kernel_size=(3, 3),
              strides=(1, 1),
              activation='relu',
              padding='same'):

    x = UpSampling2D((2, 2))(x)
    for i in range(conv_iteration):
        _filters = filters//2 if i == conv_iteration - 1 and channel_reduction \
            else filters
        x = Conv2D(_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   activation=None
                   )(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x


def get_model(input_shape, is_plot=False):
    print("Model is Encoder Decoder")
    print("\nBuilding Model Now ... \r")

    # encode
    input_img = Input(shape=input_shape)

    x = _down_block(input_img, 64, conv_iteration=2)
    x = _down_block(x, 128, conv_iteration=2)
    x = _down_block(x, 256, conv_iteration=3)
    x = _down_block(x, 512, conv_iteration=3)
    x = _down_block(x, 512, conv_iteration=3)

    # decode
    x = _up_block(x, 512, conv_iteration=3, channel_reduction=False)
    x = _up_block(x, 512, conv_iteration=3)
    x = _up_block(x, 256, conv_iteration=3)
    x = _up_block(x, 128, conv_iteration=2)
    x = _up_block(x, 64, conv_iteration=1, channel_reduction=False)

    x = Conv2D(input_shape[2], (3, 3), padding="same")(x)
    decoded = Activation("sigmoid")(x)

    autoencoder = Model(inputs=input_img,
                        outputs=decoded)

    if is_plot:
        from keras.utils.visualize_util import plot
        plot(autoencoder, to_file='model.png', show_shapes=True)

    return autoencoder


if __name__ == '__main__':
    get_model((224, 224, 3), is_plot=True)
