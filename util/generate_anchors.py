import numpy as np
import tensorflow as tf


def _whctrs(anchor):
    """
    :param anchor:
    :return: width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def tf_whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    :param ws:
    :param hs:
    :param x_ctr:
    :param y_ctr:
    :return:
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def tf_mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, tf.newaxis]
    hs = hs[:, tf.newaxis]
    anchors = tf.concat((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)), axis=1)
    return anchors


def _ratio_enum(anchor, ratios):
    """
       Enumerate a set of anchors for each aspect ratio wrt an anchor.
       """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def tf_ratio_enum(anchor, ratios):
    """
       Enumerate a set of anchors for each aspect ratio wrt an anchor.
       """

    w, h, x_ctr, y_ctr = tf_whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = tf.round(tf.sqrt(size_ratios))
    hs = tf.round(ws * ratios)
    anchors = tf_mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def tf_scale_num(anchor, scales):
    w, h, x_ctr, y_ctr = tf_whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = tf_mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size], dtype='float32') - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def tf_generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                        scales=2 ** np.arange(3, 6)):
    ratios = tf.convert_to_tensor(ratios)
    scales = tf.convert_to_tensor(scales)
    base_anchor = tf.constant([1, 1, base_size, base_size], dtype=tf.float32) - 1
    ratio_anchors = tf_ratio_enum(base_anchor, ratios)
    anchors = tf.concat([tf_scale_num(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.get_shape().as_list()[0])], axis=0)
    return anchors


if __name__ == '__main__':
    # import time
    # t = time.time()
    # a = generate_anchors()
    # print(time.time() - t)
    # print(a)
    # from IPython import embed; embed()
    anchors = tf_generate_anchors(
        16, scales=np.asarray((8, 16, 32), 'float32'),
        ratios=[0.5, 1, 2])
    print(anchors)
