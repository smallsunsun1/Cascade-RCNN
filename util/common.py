import tensorflow as tf
import numpy as np
import sys

sys.path.append("..")

from config.config import _C


def tf_filter_boxes_inside_shape(boxes, shape):
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    indices = tf.where((boxes[:, 0] >= 0) &
                       (boxes[:, 1] >= 0) &
                       (boxes[:, 2] <= w) &
                       (boxes[:, 3] <= h))[:, 0]
    return indices, tf.gather(boxes, indices)

def box_to_point8(boxes):
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape([-1, 2])
    return b

def point8_to_box(points):
    p = points.reshape([-1, 4, 2])
    minxy = p.min(axis=1)
    maxxy = p.max(axis=1)
    return np.concatenate([minxy, maxxy], axis=1)

def segmentation_to_mask(polys, height, width):
    """
    Convert polygon to binary masks
    :param polys: a list of nx2 float array
    :param height:
    :param width:
    :return: a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)

def clip_boxes(boxes, shape):
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)

def filter_boxes_inside_shape(boxes, shape):
    """
    :param boxes: nx4 float
    :param shape: (h, w)
    :return: indices: (K, ) selection: (Kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where((boxes[:, 0] >= 0) &
                       (boxes[:, 1] >= 0) &
                       (boxes[:, 2] <= w) &
                       (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]


def image_preprocess(image):
    image = tf.cast(image, tf.float32)
    mean = _C.PREPROC.PIXEL_MEAN
    std = np.asarray(_C.PREPROC.PIXEL_STD)
    mean = 127.5
    std = 255.0
    image_mean = tf.convert_to_tensor(mean, tf.float32)
    image_invstd = tf.convert_to_tensor(1.0 / std, dtype=tf.float32)
    image = (image - image_mean) * image_invstd
    return image


