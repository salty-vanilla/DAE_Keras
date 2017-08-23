from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Reshape, Activation
from keras.models import Model
from keras import backend as K


def get_model(input_shape, is_plot=False):
    print("Model is Encoder Decoder")
    print("\nBuilding Model Now ... \r")

    # encode
    input_img = Input(shape=input_shape)

    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(input_img)
    x = BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # decode
    x = UpSampling2D((2, 2))(encoded)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(input_shape[2], 3, 3, activation="relu", border_mode="same")(x)
    x = BatchNormalization()(x)
    decoded = Activation("sigmoid")(x)

    autoencoder = Model(input=input_img, output=decoded)

    if is_plot:
        from keras.utils.visualize_util import plot
        plot(autoencoder, to_file='model.png', show_shapes=True)

    return autoencoder
