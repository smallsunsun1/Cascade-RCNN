import tensorflow as tf
import itertools
import numpy as np
import sys

sys.path.append("..")

from tensorflow import keras
from custom_op.ops import group_normalization
from config.config import _C
from util.box_ops import tf_area
from util.generate_anchors import tf_generate_anchors
from .model_box import roi_align
from .model_rpn import generate_rpn_proposals, rpn_losses


def fpn_model(features):
    """

    :param features: ([tf.Tensor]): ResNet features c2-c5
    :return: [tf.Tensor]: FPN features p2-p6
    """
    num_channel = _C.FPN.NUM_CHANNEL
    use_gn = (_C.FPN.NORM == 'GN')

    def upsample2x(x):
        shape2d = tf.shape(x)[1:3]
        x = tf.image.resize_bilinear(x, 2 * shape2d, align_corners=True)
        return x

    lat_2345 = [keras.layers.Conv2D(num_channel, 1, padding="same", name='lateral_1x1_c{}'.format(i + 2))(c)
                for i, c in enumerate(features)]
    if use_gn:
        lat_2345 = [group_normalization(c) for c in lat_2345]
    lat_sum_5432 = []
    for idx, lat in enumerate(lat_2345):
        if idx == 0:
            lat_sum_5432.append(lat)
        else:
            lat = lat + upsample2x(lat_sum_5432[-1])
            lat_sum_5432.append(lat)
    p2345 = [keras.layers.Conv2D(num_channel, 3, padding='same', name='posthoc_3x3_p{}'.format(i + 2))(c)
             for i, c in enumerate(lat_sum_5432[::-1])]
    if use_gn:
        p2345 = [group_normalization(c) for c in p2345]
    p6 = keras.layers.MaxPooling2D((1, 1), 2, padding="valid", name="maxpool_p6")(p2345[-1])
    return p2345 + [p6]


def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5.
    Args:
        boxes (nx4):
    Returns:
        [tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        [tf.Tensor]: 4 tensors, the gathered boxes in each level.
    Be careful that the returned tensor could be empty.
    """
    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.cast(tf.floor(
        4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),  # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]
    level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                     for i, x in enumerate(level_ids)]
    for idx, value in enumerate(num_in_levels):
        tf.summary.scalar('num_roi_level{}'.format(idx), tensor=value)
    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


def multilevel_roi_align(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level 2-5
        rcnn_boxes (tf.Tensor): nx4 boxes
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope('roi_level{}'.format(i + 2)):
            boxes_on_featuremap = boxes * (1.0 / _C.FPN.ANCHOR_STRIDES[i])
            all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois


def multilevel_rpn_losses(
        multilevel_anchors, multilevel_label_logits, multilevel_box_logits):
    """
    Args:
        multilevel_anchors: #lvl RPNAnchors
        multilevel_label_logits: #lvl tensors of shape HxWxA
        multilevel_box_logits: #lvl tensors of shape HxWxAx4
    Returns:
        label_loss, box_loss
    """
    num_lvl = len(_C.FPN.ANCHOR_STRIDES)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope('rpn_losses'):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]
            label_loss, box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(),
                multilevel_label_logits[lvl], multilevel_box_logits[lvl])
            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2], name='label_loss')
        total_box_loss = tf.add_n(losses[1::2], name='box_loss')
        tf.summary.scalar('total_label_loss', total_label_loss)
        tf.summary.scalar('total_box_loss', total_box_loss)
    return [total_label_loss, total_box_loss]


def generate_fpn_proposals(
        multilevel_pred_boxes, multilevel_label_logits, image_shape2d, training=True):
    """
    Args:
        multilevel_pred_boxes: #lvl HxWxAx4 boxes
        multilevel_label_logits: #lvl tensors of shape HxWxA
    Returns:
        boxes: kx4 float
        scores: k logits
    """
    num_lvl = len(_C.FPN.ANCHOR_STRIDES)
    assert len(multilevel_pred_boxes) == num_lvl
    assert len(multilevel_label_logits) == num_lvl

    all_boxes = []
    all_scores = []
    if _C.FPN.PROPOSAL_MODE == 'Level':
        fpn_nms_topk = _C.RPN.TRAIN_PER_LEVEL_NMS_TOPK if training else _C.RPN.TEST_PER_LEVEL_NMS_TOPK
        for lvl in range(num_lvl):
            with tf.name_scope('Lvl{}'.format(lvl + 2)):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]

                proposal_boxes, proposal_scores = generate_rpn_proposals(
                    tf.reshape(pred_boxes_decoded, [-1, 4]),
                    tf.reshape(multilevel_label_logits[lvl], [-1]),
                    image_shape2d, fpn_nms_topk)
                all_boxes.append(proposal_boxes)
                all_scores.append(proposal_scores)

        proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
        proposal_scores = tf.concat(all_scores, axis=0)  # n
        # Here we are different from Detectron.
        # Detectron picks top-k within the batch, rather than within an image. However we do not have a batch.
        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices)
    else:
        for lvl in range(num_lvl):
            with tf.name_scope('Lvl{}'.format(lvl + 2)):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]
                all_boxes.append(tf.reshape(pred_boxes_decoded, [-1, 4]))
                all_scores.append(tf.reshape(multilevel_label_logits[lvl], [-1]))
        all_boxes = tf.concat(all_boxes, axis=0)
        all_scores = tf.concat(all_scores, axis=0)
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            all_boxes, all_scores, image_shape2d,
            _C.RPN.TRAIN_PRE_NMS_TOPK if training else _C.RPN.TEST_PRE_NMS_TOPK,
            _C.RPN.TRAIN_POST_NMS_TOPK if training else _C.RPN.TEST_POST_NMS_TOPK)

    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), \
           tf.stop_gradient(proposal_scores, name='scores')


def slice_feature_and_anchors(p23456, anchors):
    for i, stride in enumerate(_C.FPN.ANCHOR_STRIDES):
        with tf.name_scope('FPN_slice_lvl{}'.format(i)):
            anchors[i] = anchors[i].narrow_to(p23456[i])

