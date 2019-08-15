import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")

from tensorflow import keras

from custom_op import ops
from config.config import _C

# def get_norm():
#     if _C.BACKBONE.NORM == 'None':
#         return lambda x:x
#     elif _C.BACKBONE.NORM == 'GN':
#         return lambda x:ops.group_normalization(x)
#     else:
#         return lambda x:ops.batch_normalization(x)

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.shape[3]
    if n_in != n_out:
        if not _C.MODE_FPN and stride == 2:
            l = l[:, :-1, :-1, :]
        return keras.layers.Conv2D(n_out, (1, 1), strides=stride, padding="same",
                                   activation=activation, name='convshortcut')(l)
    else:
        return l


def resnet_bottleneck(l, ch_out, stride, is_train=True):
    shortcut = l
    shortcut = resnet_shortcut(shortcut, ch_out * 4, stride)
    if _C.BACKBONE.STRIDE_1X1:
        if stride == 2:
            l = l[:, :-1, :-1, :]
        l = keras.layers.Conv2D(ch_out, 1, strides=stride, name='conv1', padding='same')(l)
        l = keras.layers.Conv2D(ch_out, 3, strides=1, name='conv2', padding='same')(l)
    else:
        l = keras.layers.Conv2D(ch_out, 1, strides=1, name='conv1', padding='same')(l)
        if stride == 2:
            l = tf.pad(l, [[0, 0], [1, 0], [1, 0], [0, 0]])
            l = keras.layers.Conv2D(ch_out, 3, strides=2, padding="valid", name='conv2')(l)
        else:
            l = keras.layers.Conv2D(ch_out, 3, strides=stride, name='conv2', padding='same')(l)
    if _C.BACKBONE.NORM == 'GN':
        l = keras.layers.Conv2D(ch_out * 4, 1, name='conv3', padding='same')(l)
        l = ops.group_normalization(l)
        shortcut = ops.group_normalization(shortcut)
    if _C.BACKBONE.NORM == 'BN':
        l = keras.layers.Conv2D(ch_out * 4, 1, name='conv3', padding='same')(l)
        l = ops.batch_normalization(l, is_train)
        shortcut = ops.batch_normalization(shortcut, is_train)
    ret = l + shortcut
    return tf.nn.relu(ret, name='output')


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_c4_backbone(image, num_blocks, is_train=True):
    assert len(num_blocks) == 3
    l = tf.pad(image, [[0, 0], [3, 2], [3, 2], [0, 0]])
    l = keras.layers.Conv2D(64, 7, strides=2, padding='valid', name='conv0')(l)
    l = tf.pad(l, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l = keras.layers.MaxPooling2D((3, 3), strides=2, name='pool0')(l)
    function_block = lambda l, ch_out, stride: resnet_bottleneck(l, ch_out, stride, is_train)
    c2 = resnet_group('group0', l, function_block, 64, num_blocks[0], 1)
    c3 = resnet_group('group1', c2, function_block, 128, num_blocks[1], 2)
    c4 = resnet_group('group2', c3, function_block, 256, num_blocks[2], 2)
    # 16x downsampling up to now
    return c4

def resnet_conv5(image, num_block, is_train=True):
    function_block = lambda l, ch_out, stride: resnet_bottleneck(l, ch_out, stride, is_train)
    with tf.variable_scope('resnet_conv5', reuse=tf.AUTO_REUSE):
        l = resnet_group('group3', image, function_block, 512, num_block, 2)
    return l

def resnet_fpn_backbone(image, num_blocks, is_train=True):
    shape2d = tf.shape(image)[1:3]
    mult = float(_C.FPN.RESOLUTION_REQUIREMENT)
    new_shaped2d = tf.cast(tf.ceil(tf.cast(shape2d, tf.float32) / mult) * mult, tf.int32)
    pad_shaped2d = new_shaped2d - shape2d
    assert len(num_blocks) == 4, num_blocks
    pad_base = [3, 2]
    l = tf.pad(image, [[0, 0], [pad_base[0], pad_base[1] + pad_shaped2d[0]],
                       [pad_base[0], pad_base[1] + pad_shaped2d[1]], [0, 0]])
    l = keras.layers.Conv2D(64, 7, strides=2, name='conv0')(l)
    l = tf.pad(l, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l = keras.layers.MaxPooling2D((3, 3), 2, name='pool0')(l)
    function_block = lambda l, ch_out, stride: resnet_bottleneck(l, ch_out, stride, is_train)
    c2 = resnet_group('group0', l, function_block, 64, num_blocks[0], 1)
    c3 = resnet_group('group1', c2, function_block, 128, num_blocks[1], 2)
    c4 = resnet_group('group2', c3, function_block, 256, num_blocks[2], 2)
    c5 = resnet_group('group3', c4, function_block, 512, num_blocks[3], 2)
    # 32x downsampling up to now
    # size of c5: ceil(input/32)
    return c2, c3, c4, c5

