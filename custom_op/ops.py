import tensorflow as tf

from tensorflow import keras
from .deform_conv import tf_batch_map_offsets


def batch_normalization(x, is_train=True):
    layer = keras.layers.BatchNormalization()
    y = layer(x, is_train)
    for ele in layer.updates:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ele)
    return y


def group_normalization(x, G=32, esp=1e-5):
    shape_info = x.get_shape().as_list()
    x = tf.transpose(x, [0, 3, 1, 2])
    x_shape = tf.shape(x)
    C = x_shape[1]
    H = x_shape[2]
    W = x_shape[3]
    G = tf.minimum(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = tf.Variable(tf.ones(shape=[shape_info[3]]), dtype=tf.float32, name='gamma')
    beta = tf.Variable(tf.ones(shape=[shape_info[3]]), dtype=tf.float32, name='beta')
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])

    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
    output = tf.transpose(output, [0, 2, 3, 1])
    output.set_shape(shape_info)
    return output


def batch_conv(inp, filters):
    """
        inp 的shape为[B, H, W, channels]
        filters 的shape为[B, kernel_size, kernel_size, channels, out_channels]
    """
    filters = tf.transpose(filters, perm=[1, 2, 0, 3, 4])
    filters_shape = tf.shape(filters)
    filters = tf.reshape(filters,
                         [filters_shape[0], filters_shape[1], filters_shape[2] * filters_shape[3], filters_shape[4]])
    inp_r = tf.transpose(inp, [1, 2, 0, 3])
    inp_shape = tf.shape(inp_r)
    inp_r = tf.reshape(inp_r, [1, inp_shape[0], inp_shape[1], inp_shape[2] * inp_shape[3]])
    padding = 'VALID'
    out = tf.nn.depthwise_conv2d(inp_r, filter=filters, strides=[1, 1, 1, 1], padding=padding)
    out = tf.reshape(out, [inp_shape[0] - filters_shape[0] + 1, inp_shape[1] - filters_shape[1] + 1, inp_shape[2],
                           inp_shape[3], filters_shape[4]])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
    return out


def carafe(feature_map, cm, upsample_scale, k_encoder, kernel_size):
    """implementation os ICCV 2019 oral presentation CARAFE module"""
    static_shape = feature_map.get_shape().as_list()
    f1 = keras.layers.Conv2D(cm, (1, 1), padding="valid")(feature_map)
    encode_feature = keras.layers.Conv2D(upsample_scale * upsample_scale * kernel_size * kernel_size,
                                         (k_encoder, k_encoder), padding="same")(f1)
    encode_feature = tf.nn.depth_to_space(encode_feature, upsample_scale)
    encode_feature = tf.nn.softmax(encode_feature, axis=-1)
    """encode_feature [B x (h x scale) x (w x scale) x (kernel_size * kernel_size)]"""
    extract_feature = tf.image.extract_patches(feature_map, [1, kernel_size, kernel_size, 1],
                                                     strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
    """extract feature [B x h x w x (channel x kernel_size x kernel_size)]"""
    extract_feature = keras.layers.UpSampling2D((upsample_scale, upsample_scale))(extract_feature)
    extract_feature_shape = tf.shape(extract_feature)
    B = extract_feature_shape[0]
    H = extract_feature_shape[1]
    W = extract_feature_shape[2]
    block_size = kernel_size * kernel_size
    extract_feature = tf.reshape(extract_feature, [B, H, W, block_size, -1])
    extract_feature = tf.transpose(extract_feature, [0, 1, 2, 4, 3])
    """extract feature [B x (h x scale) x (w x scale) x channel x (kernel_size x kernel_size)]"""
    encode_feature = tf.expand_dims(encode_feature, axis=-1)
    upsample_feature = tf.matmul(extract_feature, encode_feature)
    upsample_feature = tf.squeeze(upsample_feature, axis=-1)
    if static_shape[1] is not None:
        static_shape[1] = static_shape[1] * 2
    if static_shape[2] is not None:
        static_shape[2] = static_shape[2] * 2
    upsample_feature.set_shape([static_shape[0], static_shape[1], static_shape[2], static_shape[3]])
    return upsample_feature


class DeformableConv(object):
    def __init__(self, filters, use_seperate_conv=True):
        self.filters = filters
        if use_seperate_conv:
            self.conv_layer = keras.layers.SeparableConv2D(filters=filters * 3, kernel_size=(3, 3), padding='same',
                                                            use_bias=False)
        else:
            self.conv_layer = keras.layers.Conv2D(filters=filters * 3, kernel_size=(3, 3), padding='same',
                                                         use_bias=False)

    def __call__(self, x):
        conv_res = self.conv_layer(x)
        offsets = conv_res[:, :, :, : 2 * self.filters]
        weights = conv_res[:, :, :, 2 * self.filters :]
        x_shape = tf.shape(x)
        x_shape_list = x.get_shape().as_list()
        x = self._to_bc_h_w(x, x_shape)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        weights = self._to_bc_h_w(weights, x_shape)
        x_offset = tf_batch_map_offsets(x, offsets)
        weights = tf.expand_dims(weights, axis=1)
        weights = self._to_b_h_w_c(weights, x_shape)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)
        x_offset = tf.multiply(x_offset, weights)
        x_offset.set_shape(x_shape_list)
        return x_offset

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [x_shape[0], x_shape[3], 2, x_shape[1], x_shape[2]])
        x = tf.transpose(x, [0, 1, 3, 4, 2])
        x = tf.reshape(x, [-1, x_shape[1], x_shape[2], 2])
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [-1, x_shape[1], x_shape[2]])
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(x, (-1, x_shape[3], x_shape[1], x_shape[2]))
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


if __name__ == "__main__":
    a = tf.ones(shape=[2, 64, 64, 128])
    print(carafe(a, 32, 3, 3, 3))

    # batch = tf.range(5)
    # h = tf.range(10)
    # w = tf.range(9)
    # res = tf.meshgrid(h, batch, w)
    # res = tf.stack([res[1], res[0], res[2]], axis=-1)
    # data = tf.ones(shape=[20, 10, 9, 64])
    # output = tf.gather_nd(data, res)
    # print(output)

    # a = tf.reshape(tf.range(3), [1, 1, 1, 3])
    # b = keras.layers.UpSampling2D()(a)
    # print(b)
    # b = tf.tile(a, [1, 1, 1, 3])
    # c = tf.nn.depth_to_space(b, 3)
    # print(a)
    # print(c)
