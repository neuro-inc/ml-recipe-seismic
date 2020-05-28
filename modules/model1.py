from __future__ import print_function
from __future__ import absolute_import
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras import backend as K
import tensorflow as tf

from keras.layers import Input
from keras import layers
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Activation

from keras.layers import Conv2D, Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization, Concatenate
from keras.layers import ELU, Dropout, SpatialDropout2D
from keras.models import Model
from keras.optimizers import Adam

from classification_models.resnet import ResNet34
# import Run:AI gradient accumulation
import runai.ga


def masked_mse(y_true, y):
    '''Compute the masked MSE between two tensors'''
    y_pred, mask = tf.split(y, num_or_size_splits=2, axis=-1)
    axes = (1, 2)
    loss = K.sum(K.square(y_pred - y_true) * mask, axis=axes)

    mask_sum = K.sum(mask, axis=axes)
    mask_sum_corrected = tf.where(tf.math.equal(mask_sum, 0), tf.ones_like(mask_sum), mask_sum)
    loss = loss / mask_sum_corrected
    loss = tf.where(tf.math.equal(mask_sum, 0), tf.zeros_like(loss), loss)

    loss = K.mean(loss, axis=-1)
    return loss


def masked_mse_old(y_true, y):
    y_pred, mask = tf.split(y, num_or_size_splits=2, axis=-1)
    axes = (1, 2)
    loss = K.sum(K.square(y_pred - y_true) * mask, axis=axes) / K.sum(mask, axis=axes)
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    loss = K.mean(loss, axis=-1)
    return loss


def correlation(y_true, y_pred):
    '''Compute the correlation between two tensors'''
    axes = (1, 2, 3)
    demean_a = y_true - K.mean(y_true, axis=axes, keepdims=True)
    demean_b = y_pred - K.mean(y_pred, axis=axes, keepdims=True)
    cov = K.mean(demean_a * demean_b, axis=axes, keepdims=True)
    std_a = K.sqrt(K.mean(K.square(demean_a), axis=axes, keepdims=True))
    std_b = K.sqrt(K.mean(K.square(demean_b), axis=axes, keepdims=True))
    corr = cov / (std_a * std_b)
    return corr


def masked_correlation_old(y_true, y):
    y_pred, mask = tf.split(y, num_or_size_splits=2, axis=-1)
    axes = (1, 2, 3)
    sum_mask = K.sum(mask, axis=axes, keepdims=True)
    y_true = y_true * mask
    y_pred = y_pred * mask
    demean_a = (y_true - K.sum(y_true, axis=axes, keepdims=True) / sum_mask) * mask
    demean_b = (y_pred - K.sum(y_pred, axis=axes, keepdims=True) / sum_mask) * mask
    cov = K.sum(demean_a * demean_b, axis=axes, keepdims=True) / sum_mask
    std_a = K.sqrt(K.sum(K.square(demean_a), axis=axes, keepdims=True) / sum_mask)
    std_b = K.sqrt(K.sum(K.square(demean_b), axis=axes, keepdims=True) / sum_mask)
    corr = cov / (std_a * std_b)

    corr = tf.where(tf.math.is_nan(corr), tf.ones_like(corr), corr)
    return corr


def masked_correlation(y_true, y):
    '''Compute masked correlation between two tensors'''
    y_pred, mask = tf.split(y, num_or_size_splits=2, axis=-1)
    axes = (1, 2)
    mask_sum = K.sum(mask, axis=axes, keepdims=True)
    mask_sum_corrected = tf.where(tf.math.equal(mask_sum, 0), tf.ones_like(mask_sum), mask_sum)

    y_true = y_true * mask
    y_pred = y_pred * mask
    demean_a = (y_true - K.sum(y_true, axis=axes, keepdims=True) / mask_sum_corrected) * mask
    demean_b = (y_pred - K.sum(y_pred, axis=axes, keepdims=True) / mask_sum_corrected) * mask
    cov = K.sum(demean_a * demean_b, axis=axes, keepdims=True) / mask_sum_corrected
    std_a = K.sqrt(K.sum(K.square(demean_a), axis=axes, keepdims=True) / mask_sum_corrected)
    std_b = K.sqrt(K.sum(K.square(demean_b), axis=axes, keepdims=True) / mask_sum_corrected)
    corr = cov / (std_a * std_b)

    corr = tf.where(tf.math.equal(mask_sum, 0), tf.ones_like(corr), corr)
    corr = K.mean(corr, axis=-1)
    return corr


def masked_correlation_loss(y_true, y):
    return -masked_correlation(y_true, y)


def masked_correlation_old_loss(y_true, y):
    return -masked_correlation_old(y_true, y)


def decoder_block(input_tensors, filters, stage):
    '''Defines the structure of a decoder block in the network.'''
    decoder_name_base = 'decoder_' + str(stage)
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        cat_axis = 3
    else:
        cat_axis = 1
    if isinstance(input_tensors, (tuple, list)):
        x = layers.add(input_tensors)
    else:
        x = input_tensors
    x = UpSampling2D()(x)
    x = Conv2D(filters1, (3, 3), padding='same', kernel_initializer='he_normal',
               name=decoder_name_base + '_conv1')(x)
    x = Activation('elu')(x)
    x = BatchNormalization(axis=cat_axis)(x)
    x = Conv2D(filters2, (3, 3), padding='same', kernel_initializer='he_normal',
               name=decoder_name_base + '_conv2')(x)
    x = Activation('elu')(x)
    x = BatchNormalization(axis=cat_axis)(x)
    return x


def uResNet34(input_size=None, weights=None, n_carotage=None):
    '''Definition of the model structure with a ResNet-34 decoder'''
    if K.image_data_format() == 'channels_last':
        seismic_shape = input_size + (1,)
        resnet_shape = input_size + (3,)
    else:
        seismic_shape = (1,) + input_size
        resnet_shape = (3,) + input_size

    # Init Resnet Unet
    input_resnet = Input(shape=resnet_shape)
    model = ResNet34(input_shape=resnet_shape, include_top=False, weights='imagenet')
    model(input_resnet)

    stage_2 = model.get_layer('stage2_unit1_relu1').output  # 64
    stage_3 = model.get_layer('stage3_unit1_relu1').output  # 32
    stage_4 = model.get_layer('stage4_unit1_relu1').output  # 16
    stage_5 = model.get_layer('relu1').output               # 8

    x = decoder_block(stage_5, (256, 256), stage=5)
    x = decoder_block([x, stage_4], (128, 128), stage=4)
    x = decoder_block([x, stage_3], (64, 64), stage=3)
    x = decoder_block([x, stage_2], (64, 64), stage=2)

    x = decoder_block(x, (64, 32), stage=1)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = Conv2D(n_carotage, (1, 1), kernel_initializer='he_normal')(x)

    model_resnet = Model(model.get_input_at(0), x, name='uResNet34')

    # Main model
    input_seismic = Input(shape=seismic_shape)
    input_mask = Input(shape=input_size + (n_carotage,))

    x = Conv2D(3, (1, 1))(input_seismic)
    x = model_resnet(x)
    x = Concatenate()([x, input_mask])
    model = Model([input_seismic, input_mask], x, name='main')

    if weights is not None:
        print('Load weights from', weights)
        model.load_weights(weights, by_name=True)

    optimizer = Adam()
    # wrap the Keras.Optimizer with gradient accumulation of 16 steps
    optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=16)
    model.compile(loss=masked_mse, optimizer=optimizer, metrics=[masked_correlation])
    # model.compile(loss=masked_correlation_old_loss, optimizer=optimizer, metrics=[masked_correlation_old])

    return model


def tst_masked_mse():
    """Compute and print masked MSE"""
    shape = (6, 16, 8, 3)
    a = np.random.random(shape).astype(dtype=np.float32)
    b = np.random.random(shape).astype(dtype=np.float32)
    m = np.random.randint(0, 2, shape)
    m[..., 2] = 0

    y_true = K.constant(a)
    y = K.concatenate([K.constant(b), K.constant(m)], axis=-1)
    mse_gpu = K.eval(masked_mse(y_true, y))
    print('masked gpu mse:')
    print(mse_gpu.flatten())

    mse = np.square(m * (a - b)).sum(axis=(1, 2))
    nz = m.sum(axis=(1, 2))
    nz[nz == 0] = 1
    mse = mse / nz
    mse[m.sum(axis=(1, 2)) == 0] = 0
    mse = mse.mean(axis=-1)
    print('masked mse:')
    print(mse)

    # d = K.sum(y_true, axis=(1, 2)) / K.sum(K.constant(m), axis=(1, 2))
    # print(K.eval(d))
    # print(K.eval(K.mean(d, axis=-1)))
    #
    # d = tf.where(tf.math.is_inf(d), tf.zeros_like(d), d)
    # print(K.eval(d))
    # print(K.eval(K.mean(d, axis=-1)))


def tst_masked_correlation():
    '''Compute and print masked and full correlations'''
    shape = (6, 16, 8, 3)
    a = np.random.random(shape)
    b = np.random.random(shape)
    m = np.random.randint(0, 2, shape)
    m[..., 0] = 0

    print('masked gpu correlation:')
    corr = K.eval(masked_correlation(K.constant(a), K.concatenate([K.constant(b), K.constant(m)], axis=-1)))
    print(corr.flatten())

    cor = []
    for a_, b_, m_ in zip(a, b, m):
        cor_ = []
        for j in range(m.shape[-1]):
            a__ = a_[..., j].flatten()
            b__ = b_[..., j].flatten()
            m__ = m_[..., j].flatten()
            if np.where(m__)[0].size > 0:
                cor__ = np.corrcoef(a__[np.where(m__)], b__[np.where(m__)])[0, 1]
            else:
                cor__ = 1
            cor_.append(cor__)
        cor_ = np.array(cor_).mean()
        cor.append(cor_)
    cor = np.array(cor)
    print('masked correlation:')
    print(cor)

    print('full correlation:')
    print(np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(a.reshape(len(a), -1), b.reshape(len(a), -1))]).flatten())


if __name__ == '__main__':
    # input_size = (1024, 512)
    # n_carotage = 3
    # model = uResNet34(input_size=input_size, weights=None, n_carotage=n_carotage)
    # pass

    tst_masked_correlation()
