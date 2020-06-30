#
# 2D-segmentation model
#

from __future__ import print_function
from __future__ import absolute_import
import warnings
warnings.simplefilter(action='ignore')

from keras import backend as K
import tensorflow as tf

from keras import layers
from keras.layers import Input, Activation, Conv2D, UpSampling2D, BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from classification_models.resnet import ResNet34
import runai.ga.keras  # gradient accumulation

from typing import Tuple, List
from pathlib import Path


def masked_mse(y_true, y):
    """Compute the masked MSE between two tensors"""
    y_pred, mask = tf.split(y, num_or_size_splits=2, axis=-1)
    axes = (1, 2)
    loss = K.sum(K.square(y_pred - y_true) * mask, axis=axes)

    mask_sum = K.sum(mask, axis=axes)
    mask_sum_corrected = tf.where(tf.math.equal(mask_sum, 0), tf.ones_like(mask_sum), mask_sum)
    loss = loss / mask_sum_corrected
    loss = tf.where(tf.math.equal(mask_sum, 0), tf.zeros_like(loss), loss)

    loss = K.mean(loss, axis=-1)
    return loss


def masked_correlation(y_true, y):
    """Compute masked correlation between two tensors"""
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


def decoder_block(input_tensors, filters, stage):
    """Defines the structure of a decoder block in the network"""
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


def uResNet34(input_size: Tuple = None, weights: Path = None, n_carotage: int = None) -> Model:
    """Definition of the model structure with a ResNet-34 decoder"""
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

    return model


if __name__ == '__main__':
    input_size = (480, 512)
    n_carotage = 3
    model = uResNet34(input_size=input_size, weights=None, n_carotage=n_carotage)
