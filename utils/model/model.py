from keras.layers import (
    Conv2D,
    Input,
    Flatten,
    BatchNormalization,
    Dropout,
    LeakyReLU,
    Dense
)
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import UpSampling2D
from utils.params import IMG_TARGET_ROWS, IMG_TARGET_COLS
from utils.model.metric import dice_loss, dice, bce_dice_loss


K.set_image_dim_ordering('th')


def conv_bn_relu(nb_filter, nb_row, nb_col, strides=(1, 1)):
    def f(input):
        #init="he_normal"
        conv = Conv2D(
            filters=nb_filter,
            kernel_size=(nb_row, nb_col),
            strides=strides,
            kernel_initializer="he_normal",
            padding='same')(input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)
    return f


# Dropout 0.5 as default
def build_model(optimizer=None):
    if optimizer is None:
        optimizer = Adam(lr=1e-4)

    inputs = Input((1, IMG_TARGET_ROWS, IMG_TARGET_COLS), name='main_input')
    conv1 = conv_bn_relu(32, 7, 7)(inputs)
    conv1 = conv_bn_relu(32, 3, 3)(conv1)
    pool1 = conv_bn_relu(32, 2, 2, strides=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = conv_bn_relu(64, 3, 3)(drop1)
    conv2 = conv_bn_relu(64, 3, 3)(conv2)
    pool2 = conv_bn_relu(64, 2, 2, strides=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = conv_bn_relu(128, 3, 3)(drop2)
    conv3 = conv_bn_relu(128, 3, 3)(conv3)
    pool3 = conv_bn_relu(128, 2, 2, strides=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = conv_bn_relu(256, 3, 3)(drop3)
    conv4 = conv_bn_relu(256, 3, 3)(conv4)
    pool4 = conv_bn_relu(256, 2, 2, strides=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = conv_bn_relu(512, 3, 3)(drop4)
    conv5 = conv_bn_relu(512, 3, 3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # 1 way Using conv to mimic fully connected layer
    aux = Conv2D(
        filters=1,
        kernel_size=(drop5._keras_shape[2], drop5._keras_shape[3]),
        strides=(1, 1),
        activation='sigmoid')(drop5)

    aux = Flatten(name='aux_output')(aux)

    # 2 way
    # classify
    #aux = Flatten()(conv5)
    #aux = Dense(32)(aux)
    #aux = LeakyReLU()(aux)
    #aux = Dense(16)(aux)
    #aux = LeakyReLU()(aux)
    #aux = Dense(1, activation='sigmoid', name='aux_output')(aux)

    up6 = concatenate([UpSampling2D()(drop5), conv4], axis=1)
    conv6 = conv_bn_relu(256, 3, 3)(up6)
    conv6 = conv_bn_relu(256, 3, 3)(conv6)
    drop6 = Dropout(0.4)(conv6)

    up7 = concatenate([UpSampling2D()(drop6), conv3], axis=1)
    conv7 = conv_bn_relu(128, 3, 3)(up7)
    conv7 = conv_bn_relu(128, 3, 3)(conv7)
    drop7 = Dropout(0.4)(conv7)

    up8 = concatenate([UpSampling2D()(drop7), conv2], axis=1)
    conv8 = conv_bn_relu(64, 3, 3)(up8)
    conv8 = conv_bn_relu(64, 3, 3)(conv8)
    drop8 = Dropout(0.4)(conv8)

    up9 = concatenate([UpSampling2D()(drop8), conv1], axis=1)
    conv9 = conv_bn_relu(32, 3, 3)(up9)
    conv9 = conv_bn_relu(32, 3, 3)(conv9)
    drop9 = Dropout(0.4)(conv9)

    conv10 = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation='sigmoid',
        name='main_output')(drop9)

    model = Model(inputs=inputs, outputs=[conv10, aux])
    model.compile(optimizer=optimizer,
                  loss={'main_output': dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice, 'aux_output': 'acc'})
                  #loss_weights={'main_output': 1, 'aux_output': 0.5}

    return model
