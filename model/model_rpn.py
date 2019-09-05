import tensorflow as tf
import sys

sys.path.append("..")

from tensorflow import keras
from config.config import _C
from .model_box import clip_boxes


def rpn_head(feature_map, channel, num_anchors):
    """

    :param feature_map:
    :param channel:
    :param num_anchors:
    :return: label_logits: fHxfWxNA
              box_logits: fHxfWxNAx4
    """
    hidden = keras.layers.Conv2D(channel, (3, 3), activation=tf.nn.relu, padding="same")(feature_map)
    label_logits = keras.layers.Conv2D(num_anchors, (1, 1), padding="same", name='class')(hidden)
    box_logits = keras.layers.Conv2D(4 * num_anchors, (1, 1), padding="same", name='box')(hidden)
    label_logits = tf.squeeze(label_logits)
    shp = tf.shape(box_logits)
    box_logits = tf.reshape(box_logits, [shp[1], shp[2], num_anchors, 4])  # fHxfWxNAx4
    return label_logits, box_logits


def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
    """

    :param anchor_labels: fHxfWxNA
    :param anchor_boxes: fHxfWxNAx4, encoded
    :param label_logits: fHxfWxNA
    :param box_logits: fHxfWxNAx4
    :return: label_loss, box_loss
    """
    valid_mask = tf.not_equal(anchor_labels, - 1)
    pos_mask = tf.equal(anchor_labels, 1)
    nr_valid = tf.count_nonzero(valid_mask, dtype=tf.int32)
    nr_pos = tf.count_nonzero(pos_mask, dtype=tf.int32)
    valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)
    valid_label_prob = tf.nn.sigmoid(valid_label_logits)
    for th in [0.5, 0.2, 0.1]:
        valid_prediction = tf.cast(tf.greater(valid_label_prob, th), tf.int32)
        nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
        pos_prediction_corr = tf.count_nonzero(tf.logical_and(tf.greater(valid_label_prob, th),
                                                              tf.equal(valid_prediction, valid_anchor_labels)),
                                               dtype=tf.int32)
        recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
        placeholder = 0.5
        recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name="recall_th{}".format(th))
        precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
        precision = tf.where(tf.equal(nr_pos_prediction, 0),
                             placeholder, precision, name='precision_th{}'.format(th))
        tf.summary.scalar('precision_th{}'.format(th), precision)
        tf.summary.scalar('recall_th{}'.format(th), recall)
    placeholder = 0.0
    label_loss = tf.losses.sigmoid_cross_entropy(tf.cast(valid_anchor_labels, tf.float32),
                                                 valid_label_logits, reduction=tf.losses.Reduction.SUM)
    label_loss = label_loss / _C.RPN.BATCH_PER_IM
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name="label_loss")
    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(pos_anchor_boxes, pos_box_logits, delta=delta,
                                    reduction=tf.losses.Reduction.SUM) / delta
    box_loss = box_loss * (1. / _C.RPN.BATCH_PER_IM)
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss)
    return [label_loss, box_loss]


def generate_rpn_proposals(boxes, scores, img_shape, pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output
    :param boxes: nx4 float dtype, the proposal boxes. Decoded to floatbox already
    :param scores: n float, the logits
    :param img_shape: [h, w]
    :param pre_nms_topk: post_nms_topk (int): See above.
    :param post_nms_topk:
    :return:
        boxes: kx4 float
        scores: k logits
    """
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk
    topk = tf.minimum(pre_nms_topk, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)
    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(tf.greater(wbhb, _C.RPN.MIN_SIZE), axis=1)  # n
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)
    # TODO not needed
    topk_valid_boxes_y1x1y2x2 = tf.reshape(tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
                                           [-1, 4], name="nms_input_boxes")
    nms_indices = tf.image.non_max_suppression(topk_valid_boxes_y1x1y2x2,
                                               topk_valid_scores, max_output_size=post_nms_topk,
                                               iou_threshold=_C.RPN.PROPOSAL_NMS_THRESH)
    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    proposal_boxes = tf.gather(topk_valid_boxes, nms_indices)
    proposal_scores = tf.gather(topk_valid_scores, nms_indices)
    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name="boxes"), tf.stop_gradient(proposal_scores, name="scores")
