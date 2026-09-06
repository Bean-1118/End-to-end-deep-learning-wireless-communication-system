import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, Flatten, Activation, BatchNormalization, AveragePooling2D, MaxPooling2D, Lambda, Reshape, Concatenate, Add, Multiply, Cropping1D
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
M = 128
batch_size = 512
block_length = M
len_h = 8
EbNo_train = 20
R = 0.5
std = np.sqrt(1 / (2 * R * EbNo_train))
len_info = 40

def convolution(x, h):
    """
    Original circular multipath convolution.

    x shape:
        (batch, block_length + len_h, 1)

    h shape:
        (batch, len_h, 1)
    """
    y = x * tf.cast(tf.reshape(h[:, 0, 0], [-1, 1, 1]), tf.float32)
    for i in range(1, len_h):
        cur = x * tf.cast(tf.reshape(h[:, i, 0], [-1, 1, 1]), tf.float32)
        cur = tf.concat([cur[:, -i:, :], cur[:, :-i, :]], axis=1)
        y = y + cur
    return y

def bilinear_mul(x, info):
    """
    Original bilinear multiplication operation.

    x:
        (batch, block_length + len_h, 2)

    info:
        (batch, len_info)
    """
    x_reshape = tf.reshape(x, [-1, (block_length + len_h) * 2, 1])
    info_reshape = tf.reshape(info, [-1, 1, len_info])
    mat_reshape = tf.matmul(x_reshape, info_reshape)
    output = tf.reshape(mat_reshape, [-1, block_length + len_h, len_info * 2])
    return output

class WirelessChannelLayer(tf.keras.layers.Layer):
    """
    Keras 3 wrapper for the original channel operations.

    The following original operations are preserved:

    1. Encoder-output L2 normalisation
    2. Zero padding
    3. Complex multipath convolution
    4. AWGN addition
    """

    def __init__(self, configured_batch_size, configured_block_length, configured_len_h, **kwargs):
        super().__init__(**kwargs)
        self.configured_batch_size = configured_batch_size
        self.configured_block_length = configured_block_length
        self.configured_len_h = configured_len_h

    def call(self, inputs):
        (x, input_h, input_sdv) = inputs
        normalisation_scale = tf.math.sqrt(tf.cast(self.configured_batch_size * self.configured_block_length, tf.float32))
        layer_4_normalized = tf.math.scalar_mul(normalisation_scale, tf.math.l2_normalize(x))
        x_pad = tf.pad(layer_4_normalized, tf.constant([[0, 0], [0, self.configured_len_h], [0, 0]]))
        h_r = tf.reshape(input_h[:, :, 0], [-1, self.configured_len_h, 1])
        h_i = tf.reshape(input_h[:, :, 1], [-1, self.configured_len_h, 1])
        x_r = tf.reshape(x_pad[:, :, 0], [-1, self.configured_block_length + self.configured_len_h, 1])
        x_i = tf.reshape(x_pad[:, :, 1], [-1, self.configured_block_length + self.configured_len_h, 1])
        o_r = convolution(x_r, h_r) - convolution(x_i, h_i)
        o_i = convolution(x_r, h_i) + convolution(x_i, h_r)
        output = tf.concat([o_r, o_i], axis=-1)
        stddev = tf.reshape(input_sdv, [-1, 1, 1])
        unit_noise = tf.random.normal(shape=tf.shape(output), mean=0.0, stddev=1.0, dtype=tf.float32)
        output_noisy = output + unit_noise * stddev
        return output_noisy

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.configured_block_length + self.configured_len_h, 2)

    def get_config(self):
        config = super().get_config()
        config.update({'configured_batch_size': self.configured_batch_size, 'configured_block_length': self.configured_block_length, 'configured_len_h': self.configured_len_h})
        return config

class BilinearMulLayer(tf.keras.layers.Layer):
    """
    Keras 3 wrapper for the original bilinear_mul() function.
    """

    def call(self, inputs):
        (x, estimated_channel) = inputs
        return bilinear_mul(x, estimated_channel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], block_length + len_h, len_info * 2)

def end_to_end():
    input_shape = (M, 1)
    input_h_shape = (len_h, 2)
    input_sdv_shape = (1,)
    input_bits = Input(shape=input_shape, name='input_bits')
    input_h = Input(shape=input_h_shape, name='input_channel')
    input_sdv = Input(shape=input_sdv_shape, name='input_noise_std')
    x = Conv1D(256, 5, padding='same', activation='relu', name='encoder_conv_1')(input_bits)
    x = Conv1D(128, 3, padding='same', activation='relu', name='encoder_conv_2')(x)
    x = Conv1D(64, 3, padding='same', activation='relu', name='encoder_conv_3')(x)
    x = Conv1D(2, 3, padding='same', activation='relu', name='encoder_conv_4')(x)
    output_noisy = WirelessChannelLayer(configured_batch_size=batch_size, configured_block_length=block_length, configured_len_h=len_h, name='wireless_channel')([x, input_h, input_sdv])
    x = Conv1D(256, 5, padding='same', activation='relu', name='channel_est_conv_1')(output_noisy)
    x = Conv1D(128, 3, padding='same', activation='relu', name='channel_est_conv_2')(x)
    x = Conv1D(64, 3, padding='same', activation='relu', name='channel_est_conv_3')(x)
    x = Conv1D(32, 3, padding='same', activation='relu', name='channel_est_conv_4')(x)
    x = Conv1D(3, 3, padding='same', activation='relu', name='channel_est_conv_5')(x)
    conv4_flat = Reshape((3 * (block_length + len_h),), name='channel_est_flatten')(x)
    x = Dense(100, activation='linear', name='channel_est_dense_1')(conv4_flat)
    estimated_channel = Dense(len_info, activation='linear', name='estimated_channel')(x)
    x_combine = BilinearMulLayer(name='bilinear_multiplication')([output_noisy, estimated_channel])
    x = Conv1D(256, 5, padding='same', activation='relu', name='decoder_conv_1')(x_combine)
    x_ori_1 = Conv1D(128, 5, padding='same', activation='relu', name='decoder_res1_input')(x)
    x = Conv1D(128, 5, padding='same', activation='relu', name='decoder_res1_conv1')(x_ori_1)
    x = Conv1D(128, 5, padding='same', name='decoder_res1_conv2')(x)
    x = Add(name='decoder_res1_add')([x, x_ori_1])
    x = Activation('relu', name='decoder_res1_relu')(x)
    x_ori_2 = Conv1D(64, 5, padding='same', activation='relu', name='decoder_res2_input')(x)
    x = Conv1D(64, 5, padding='same', activation='relu', name='decoder_res2_conv1')(x_ori_2)
    x = Conv1D(64, 3, padding='same', name='decoder_res2_conv2')(x)
    x = Add(name='decoder_res2_add')([x, x_ori_2])
    x = Activation('relu', name='decoder_res2_relu')(x)
    x = Conv1D(32, 3, padding='same', activation='relu', name='decoder_conv_final')(x)
    pred_logits = Conv1D(1, 3, padding='same', name='pred_logits_full')(x)
    pred_prob = Activation('sigmoid', name='pred_prob_full')(pred_logits)
    pred_logits_reshape = Cropping1D(cropping=(0, len_h), name='pred_logits')(pred_logits)
    pred_prob_reshape = Cropping1D(cropping=(0, len_h), name='pred_prob')(pred_prob)
    model = Model(inputs=[input_bits, input_h, input_sdv], outputs=[pred_logits_reshape, pred_prob_reshape], name='end_to_end_wireless_model')
    return model
