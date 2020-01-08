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
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths
    #    lt = tf.maximum(boxlist1[:, tf.newaxis, :2], boxlist2[tf.newaxis, :, :2])
    #    rb = tf.minimum(boxlist1[:, tf.newaxis, 2:], boxlist2[tf.newaxis, :, 2:])
    #    wh = tf.maximum(rb - lt + 1, 0)
    #    overlap = wh[:, :, 0] * wh[:, :, 1]
    #    return overlap

def tf_iou(boxlist1, boxlist2):
    areas1 = tf_area(boxlist1)
    areas2 = tf_area(boxlist2)
    intersections = pairwise_intersection(boxlist1, boxlist2)
    intersections = tf.maximum(intersections, 0)
    unions = (
            tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

def tf_ioa(boxes1, boxes2):
    intersect = pairwise_intersection(boxes1, boxes2)
    inv_areas = tf.expand_dims(1.0 / tf_area(boxes2), axis=0)
    return intersect * inv_areas

