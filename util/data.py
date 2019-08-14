import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")

from config.config import _C
from .generate_anchors import tf_generate_anchors
from .box_ops import tf_area, tf_ioa, tf_iou
from .common import tf_filter_boxes_inside_shape


def tf_get_all_anchors(stride=None, sizes=None, ratios=(0.5, 1, 2)):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors
    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.
    """
    if stride is None:
        stride = _C.RPN.ANCHOR_STRIDE
    if sizes is None:
        sizes = _C.RPN.ANCHOR_SIZES
    cell_anchors = tf_generate_anchors(stride, scales=np.asarray(sizes, dtype=np.float32) / stride,
                                       ratios=np.asarray(ratios, dtype=np.float32))
    PREPROC_MAX_SIZE = 1333
    max_size = PREPROC_MAX_SIZE
    field_size = int(np.ceil(max_size / stride))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    shifts = shifts.astype(np.float32)
    K = shifts.shape[0]
    A = tf.shape(cell_anchors)[0]
    field_of_anchors = (
            tf.reshape(cell_anchors, shape=[1, A, 4]) + tf.transpose(tf.reshape(shifts, [1, K, 4]), perm=[1, 0, 2]))
    field_of_anchors = tf.reshape(field_of_anchors, shape=[field_size, field_size, A, 4])
    field_of_anchors = tf.cast(field_of_anchors, tf.float32)
    s1, s2, s3, s4 = tf.split(field_of_anchors, num_or_size_splits=4, axis=-1)
    s3 = s3 + 1
    s4 = s4 + 1
    return tf.concat([s1, s2, s3, s4], axis=-1)


def tf_get_all_anchors_fpn(strides=(4, 8, 16, 32, 64), sizes=(32, 64, 128, 256, 512), ratios=(0.5, 1, 2)):
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = tf_get_all_anchors(stride, (size,), ratios)
        foas.append(foa)
    return foas


@tf.function
def tf_get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
        Label each anchor as fg/bg/ignore.
        Args:
            anchors: Ax4 float
            gt_boxes: Bx4 float, non-crowd
            crowd_boxes: Cx4 float
        Returns:
            anchor_labels: (A,) int. Each element is {-1, 0, 1}
            anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    N = tf.shape(crowd_boxes)[0]

    def filter_box_label(labels, value, max_num):
        curr_inds = tf.where(tf.equal(labels, value))
        length = tf.shape(curr_inds)[0]

        def true_fn(curr_inds, labels, max_num):
            curr_inds = tf.random.shuffle(curr_inds)
            res_inds = curr_inds[max_num:]
            labels -= tf.scatter_nd(indices=tf.cast(res_inds, tf.int32), updates=(tf.gather_nd(labels, res_inds) + 1),
                                    shape=tf.shape(labels))
            return curr_inds[:max_num], labels

        def false_fn(curr_inds, labels):
            return curr_inds, labels

        a, b = tf.cond(length > max_num, lambda: true_fn(curr_inds, labels, max_num),
                       lambda: false_fn(curr_inds, labels))
        return a, b

    NA = tf.shape(anchors)[0]
    box_ious = tf_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = tf.argmax(box_ious, axis=1)  # NA
    ious_max_per_anchor = tf.reduce_max(box_ious, axis=1)  # NA
    ious_max_per_gt = tf.reduce_max(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = tf.where(tf.equal(box_ious, ious_max_per_gt))[:, 0]  # NA
    anchor_labels = -tf.ones(shape=[NA, ], dtype=tf.int32)
    temp = tf.unique(anchors_with_max_iou_per_gt).y
    anchor_labels += tf.scatter_nd(indices=tf.cast(tf.expand_dims(temp, -1), tf.int32),
                                   updates=tf.fill(tf.shape(temp), 2), shape=tf.shape(anchor_labels))
    anchor_labels = tf.where(ious_max_per_anchor >= _C.RPN.POSITIVE_ANCHOR_THRESH, tf.ones_like(anchor_labels),
                             anchor_labels)
    anchor_labels = tf.where(ious_max_per_anchor < _C.RPN.NEGATIVE_ANCHOR_THRESH, tf.zeros_like(anchor_labels),
                             anchor_labels)

    def true_fn(anchor_labels, crowd_boxes):
        cand_inds = tf.where(tf.greater_equal(anchor_labels, 0))
        cand_anchors = tf.gather_nd(anchors, cand_inds)
        ioas = tf_ioa(crowd_boxes, cand_anchors)
        overlap_with_crowd = tf.gather_nd(cand_inds,
                                          tf.where(tf.greater(tf.reduce_max(ioas, 0), _C.RPN.CROWD_OVERLAP_THRESH)))
        value = tf.scatter_nd(indices=tf.cast(tf.expand_dims(overlap_with_crowd, axis=-1), tf.int32),
                              updates=tf.cast((tf.gather(anchor_labels, overlap_with_crowd) + 1), tf.int32),
                              shape=tf.shape(anchor_labels))
        anchor_labels -= value
        return anchor_labels
        # Subsample fg labels: ignore some fg if fg is too many

    anchor_labels = tf.cond(N > 0, lambda: true_fn(anchor_labels, crowd_boxes),
                            lambda: anchor_labels)
    target_num_fg = int(_C.RPN.BATCH_PER_IM * _C.RPN.FG_RATIO)
    fg_inds, anchor_labels = filter_box_label(anchor_labels, 1, target_num_fg)
    old_num_bg = tf.reduce_sum(tf.cast(tf.equal(anchor_labels, 0), tf.int32))
    target_num_bg = _C.RPN.BATCH_PER_IM - tf.size(fg_inds)
    _, anchor_labels = filter_box_label(anchor_labels, 0, target_num_bg)
    anchor_boxes = tf.zeros(shape=[NA, 4])
    fg_boxes = tf.gather(gt_boxes, tf.gather(ious_argmax_per_anchor, fg_inds))
    anchor_boxes += tf.scatter_nd(indices=tf.cast(tf.expand_dims(fg_inds, -1), tf.int32),
                                  updates=fg_boxes, shape=tf.shape(anchor_boxes))
    return anchor_labels, anchor_boxes


def tf_get_rpn_anchor_input(im, boxes, is_crowd):
    """
        Args:
            im: an image
            boxes: nx4, floatbox, gt. shoudn't be changed
            is_crowd: n,
        Returns:
            The anchor labels and target boxes for each pixel in the featuremap.
            fm_labels: fHxfWxNA
            fm_boxes: fHxfWxNAx4
            NA will be NUM_ANCHOR_SIZES x NUM_ANCHOR_RATIOS
    """
    all_anchors = tf_get_all_anchors()
    featuremap_anchors_flatten = tf.reshape(all_anchors, [-1, 4])
    inside_ind, inside_anchors = tf_filter_boxes_inside_shape(featuremap_anchors_flatten, tf.shape(im)[:2])
    anchor_labels, anchor_gt_boxes = tf_get_anchor_labels(inside_anchors,
                                                          tf.gather_nd(boxes, tf.where(tf.equal(is_crowd, 0))),
                                                          tf.gather_nd(boxes, tf.where(tf.equal(is_crowd, 1))))
    anchorH = tf.shape(all_anchors)[0]
    anchorW = tf.shape(all_anchors)[1]
    featuremap_labels = -tf.ones((anchorH * anchorW * _C.RPN.NUM_ANCHOR,), dtype=tf.int32)
    featuremap_labels += (tf.scatter_nd(tf.cast(tf.expand_dims(inside_ind, axis=-1), tf.int32),
                                        updates=(anchor_labels + tf.ones_like(anchor_labels)),
                                        shape=tf.shape(featuremap_labels)))
    featuremap_labels = tf.reshape(featuremap_labels, (anchorH, anchorW, _C.RPN.NUM_ANCHOR))
    featuremap_boxes = tf.zeros((anchorH * anchorW * _C.RPN.NUM_ANCHOR, 4), dtype=tf.float32)
    featuremap_boxes += tf.scatter_nd(tf.cast(tf.expand_dims(inside_ind, axis=-1), tf.int32), updates=anchor_gt_boxes,
                                      shape=tf.shape(featuremap_boxes))
    return featuremap_labels, featuremap_boxes


def tf_get_multilevel_rpn_anchor_input(im, boxes, is_crowd):
    """

    :param im: an image
    :param boxes: nx4, floatbox, gt. shoun't be changed
    :param is_crowd: n
    :return:
        [(fm_labels, fm_boxes)]: Returns a tuple for each FPN level.
        Each tuple contains the anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWx NUM_ANCHOR_RATIOS
        fm_boxes: fHxfWx NUM_ANCHOR_RATIOS x4
    """
    anchors_per_level = tf_get_all_anchors_fpn()
    flatten_anchors_per_level = [tf.reshape(k, [-1, 4]) for k in anchors_per_level]
    all_anchors_flatten = tf.concat(flatten_anchors_per_level, axis=0)
    inside_ind, inside_anchors = tf_filter_boxes_inside_shape(all_anchors_flatten, tf.shape(im)[:2])
    anchor_labels, anchor_gt_boxes = tf_get_anchor_labels(inside_anchors,
                                                          tf.gather_nd(boxes, tf.where(tf.equal(is_crowd, 0))),
                                                          tf.gather_nd(boxes, tf.where(tf.equal(is_crowd, 1))))
    num_all_anchors = tf.shape(all_anchors_flatten)[0]
    all_labels = -tf.ones([num_all_anchors, ], dtype=tf.int32)
    all_labels += tf.scatter_nd(indices=tf.to_int32(tf.expand_dims(inside_ind, axis=-1)), updates=(anchor_labels + 1),
                                shape=tf.shape(all_labels))
    all_boxes = tf.zeros(shape=[num_all_anchors, 4])
    all_boxes += tf.scatter_nd(indices=tf.to_int32(tf.expand_dims(inside_ind, axis=-1)), updates=anchor_gt_boxes,
                               shape=tf.shape(all_boxes))
    start = 0
    multilevel_inputs = []
    for level_anchor in anchors_per_level:
        anchor_shape = tf.shape(level_anchor)[:3]  # FHxFWxNUM_ANCHOR_RATIOS
        num_anchor_this_level = tf.reduce_prod(anchor_shape)
        end = start + num_anchor_this_level
        multilevel_inputs.append((tf.reshape(all_labels[start:end], anchor_shape),
                                  tf.reshape(all_boxes[start:end, :], tf.concat([anchor_shape, [4, ]], axis=-1))))
        start = end
    return multilevel_inputs


