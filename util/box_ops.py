import tensorflow as tf

def tf_area(boxes):
    """
    :param boxes: nx4 floatbox
    :return: n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min + 1) * (x_max - x_min + 1), axis=[1])


def pairwise_intersection(boxlist1, boxlist2):
    """
    :param boxlist1: Nx4 floatbox
    :param boxlist2: Mx4 floatbox
    :return: a tensor with shape [N, M] representing pairwise intersection
    """
    lt = tf.maximum(boxlist1[:, tf.newaxis, :2], boxlist2[tf.newaxis, :, :2])
    rb = tf.minimum(boxlist1[:, tf.newaxis, 2:], boxlist2[tf.newaxis, :, 2:])
    wh = tf.maximum(rb - lt + 1, 0)
    overlap = wh[:, :, 0] * wh[:, :, 1]
    return overlap

def tf_iou(boxlist1, boxlist2):
    areas1 = tf_area(boxlist1)
    areas2 = tf_area(boxlist2)
    intersections = pairwise_intersection(boxlist1, boxlist2)
    intersections = tf.maximum(intersections, 0)
    unions = (
            tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return intersections / unions

def tf_ioa(boxes1, boxes2):
    intersect = pairwise_intersection(boxes1, boxes2)
    inv_areas = tf.expand_dims(1.0 / tf_area(boxes2), axis=0)
    return intersect * inv_areas

