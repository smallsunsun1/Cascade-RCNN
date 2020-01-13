import tensorflow as tf
import sys

sys.path.append("..")

from tensorflow import keras
from tensorflow.contrib.framework import sort
from config.config import _C
from util.box_ops import tf_iou
from custom_op.ops import group_normalization
from .model_box import decode_bbox_target, encode_bbox_target


class BoxProposals():
    """
        A structure to manage box proposals and their labels
    """

    def __init__(self, boxes, labels=None, fg_inds_wrt_gt=None):
        """
        Args:
            boxes: Nx4
            labels: N, each in [0, #class), the true label for each input box
            fg_inds_wrt_gt: #fg, each in [0, M)
        The last four arguments could be None when not training.
        """
        self.boxes = boxes
        self.labels = labels
        self.fg_inds_wrt_gt = fg_inds_wrt_gt

    def fg_inds(self):
        return tf.reshape(tf.where(self.labels > 0), [-1], name="fg_inds")

    def fg_boxes(self):
        return tf.gather(self.boxes, self.fg_inds(), name="fg_boxes")

    def fg_labels(self):
        return tf.gather(self.labels, self.fg_inds(), name="fg_inds")


def proposal_metrics(iou):
    """
    Add summaries for RPN proposals
    find best roi for each gt, for summary only
    :param iou: nxm, #proposal x gt
    :return:
    """
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou)
    tf.summary.scalar('mean_best_iou', mean_best_iou)
    for th in [0.3, 0.5]:
        recall = tf.truediv(tf.cast(tf.count_nonzero(tf.greater_equal(best_iou, th)), tf.float32),
                            tf.cast(tf.size(best_iou, out_type=tf.int32), tf.float32), name='recall_iou{}'.format(th))
        tf.summary.scalar(name='recall_iou{}'.format(th), tensor=recall)


def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
        Sample some boxes from all proposals for training.
        #fg is guaranteed to be > 0, because ground truth boxes will be added as proposals.
        Args:
            boxes: nx4 region proposals, floatbox
            gt_boxes: mx4, floatbox
            gt_labels: m, int32
        Returns:
            A BoxProposals instance.
            sampled_boxes: tx4 floatbox, the rois
            sampled_labels: t int32 labels, in [0, #class). Positive means foreground.
            fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
                It contains the matching GT of each foreground roi.
        """
    iou = tf_iou(boxes, gt_boxes)  # nxm
    iou_shape = tf.shape(iou)
    #print_op = tf.print("iou_shape: ", iou_shape)
    proposal_metrics(iou)
    #with tf.control_dependencies([print_op]):
    boxes = tf.concat([boxes, gt_boxes], axis=0)
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)
    # proposal=n+m from now on
    def sample_fg_bg(iou):
        fg_mask = tf.cond(tf.shape(iou)[1] > 0,
                          lambda: tf.reduce_max(iou, axis=1) >= _C.FRCNN.FG_THRESH,
                          lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))
        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        fg_inds_shape = tf.shape(fg_inds)
        #print_op = tf.print("fg_inds_shape: ", fg_inds_shape)
        num_fg = tf.minimum(int(
            _C.FRCNN.BATCH_PER_IM * _C.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        #with tf.control_dependencies([print_op]):
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            _C.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]
        tf.summary.scalar("num_fg", num_fg)
        tf.summary.scalar("num_bg", num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg, bg indices w.r.t proposals
    best_iou_ind = tf.cond(tf.shape(iou)[1] > 0,
                           lambda: tf.argmax(iou, axis=1),  # #proposal, each in 0~m-1
                           lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.int64))
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)  # num_fg
    all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)
    ret_labels = tf.concat([tf.gather(gt_labels, fg_inds_wrt_gt), tf.zeros_like(bg_inds, dtype=tf.int32)], axis=0)
    # stop the gradient -- they are meant to be training targets
    return BoxProposals(
        tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'),
        tf.stop_gradient(ret_labels, name='sampled_labels'),
        tf.stop_gradient(fg_inds_wrt_gt))


def fastrcnn_outputs(feature, num_classes, class_agnostic_regression=False):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to N x 1 x 4
    Returns:
        cls_logits: N x num_class classification logits
        reg_logits: N x num_class x 4 or Nx1x4 if class agnostic
    """
    classification = keras.layers.Dense(num_classes, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
        feature)
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = keras.layers.Dense(num_classes_for_box * 4,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.001))(feature)
    box_regression = tf.reshape(box_regression, [-1, num_classes_for_box, 4], name='output_box')
    return classification, box_regression


def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    :param labels: n,
    :param label_logits: nxC
    :param fg_boxes: nfgx4, encoded
    :param fg_box_logits: nfgxCx4 or nfgx1x4 if class agnostic
    :return: label_loss, box_loss
    """
    label_loss = tf.losses.sparse_softmax_cross_entropy(labels, label_logits)
    fg_inds = tf.where(tf.greater(labels, 0))[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int32)
    empty_fg = tf.equal(num_fg, 0)
    dim_1 = tf.shape(fg_box_logits)[1]
    cond = tf.greater(dim_1, 1)
    indices = tf.stack([tf.range(num_fg), fg_labels], axis=1)  # fg x 2
    fg_box_logits = tf.cond(cond, lambda: tf.gather_nd(fg_box_logits, indices),
                            lambda: tf.reshape(fg_box_logits, [-1, 4]))
    prediction = tf.argmax(label_logits, axis=1, name='label_prediction', output_type=tf.int32)
    correct = tf.cast(tf.equal(prediction, labels), tf.float32)
    accuracy = tf.reduce_mean(correct, name="accuracy")
    fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
    num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int32), name="num_zero")
    false_negative = tf.where(
        empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32), name='false_negative')
    fg_accuracy = tf.where(
        empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')
    diff_boxes = tf.abs(fg_boxes - fg_box_logits)
    diff_boxes_loss = tf.reduce_sum(diff_boxes, axis=-1)
    #invalid_fg_indices = tf.where(tf.math.is_inf(fg_boxes))
    #with tf.control_dependencies([invalid_fg_indices]):
    #tf.print(invalid_fg_indices)
    #diff_boxes_loss = tf.losses.huber_loss(fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.NONE)
    #diff_boxes_loss = tf.reduce_sum(diff_boxes_loss, axis=-1)
    valid_diff_boxes_loss = tf.where(tf.is_nan(diff_boxes_loss), tf.zeros_like(diff_boxes_loss), diff_boxes_loss)
    box_loss = tf.reduce_sum(valid_diff_boxes_loss)
    box_loss = tf.truediv(
        box_loss, tf.maximum(tf.cast(tf.shape(labels)[0], tf.float32), 1.0), name='box_loss')
    tf.summary.scalar('label_loss', label_loss)
    tf.summary.scalar('box_loss', box_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('fg_accuracy', fg_accuracy)
    tf.summary.scalar('false_negtive', false_negative)
    tf.summary.scalar('num_fg_label', tf.cast(num_fg, tf.float32))
    #print_op = tf.print({'frcnn_box_loss': box_loss, 'frcnn_label_loss': label_loss,
    #                     'fg_boxes': fg_boxes, 'fg_box_logits': fg_box_logits})
    #with tf.control_dependencies([print_op]):
    return [tf.identity(label_loss), tf.identity(box_loss)]


def fastrcnn_predictions_v2(boxes, score):
    """
    Generate final results from predictions of all proposals.
    :param boxes: nxclassx4 float32
    :param score:  nxclass
    :return:
        boxes:
        scores:
        labels:
    """
    boxes = tf.expand_dims(boxes[:, 1:, :], axis=0)
    score = tf.expand_dims(score[:, 1:], axis=0)
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(boxes, score,
                                                                                                       100, 1000,
                                                                                                       iou_threshold=_C.TEST.FRCNN_NMS_THRESH,
                                                                                                       score_threshold=_C.TEST.RESULT_SCORE_THRESH,
                                                                                                       clip_boxes=False)
    nmsed_classes = tf.add(nmsed_classes, 1)
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def fastrcnn_predictions(boxes, scores):
    """
        Generate final results from predictions of all proposals.
        Args:
            boxes: n#classx4 floatbox in float32
            scores: nx#class
        Returns:
            boxes: Kx4
            scores: K
            labels: K
        """
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes
        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob, out_type=tf.int32)
        ids = tf.reshape(tf.where(tf.greater(prob, _C.TEST.RESULT_SCORE_THRESH)), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        selection = tf.image.non_max_suppression(box, prob, _C.TEST.RESULT_PER_IM, _C.TEST.FRCNN_NMS_THRESH)
        selection = tf.gather(ids, selection)
        sorted_selection = sort(selection)
        mask = tf.scatter_nd(tf.cast(tf.expand_dims(sorted_selection, axis=1), tf.int32),
                             tf.ones_like(sorted_selection, dtype=tf.bool),
                             shape=output_shape)
        return mask

    masks = tf.map_fn(f, (scores, boxes), dtype=tf.bool, parallel_iterations=10)
    selected_indices = tf.where(masks)
    scores = tf.boolean_mask(scores, masks)
    topk_scores, topk_indices = tf.nn.top_k(scores, tf.minimum(_C.TEST.RESULT_PER_IM, tf.size(scores)),
                                            sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    cat_ids, box_ids = tf.unstack(filtered_selection, axis=1)
    final_scores = tf.identity(topk_scores, name='scores')
    final_labels = tf.add(cat_ids, 1, name='labels')
    final_ids = tf.stack([cat_ids, box_ids], axis=1, name='all_ids')
    final_boxes = tf.gather_nd(boxes, final_ids, name='boxes')
    return final_boxes, final_scores, final_labels


def fastrcnn_2fc_head(feature):
    dim = _C.FPN.FRCNN_FC_HEAD_DIM
    feature = keras.layers.GlobalAveragePooling2D()(feature)
    hidden = keras.layers.Dense(dim, activation=tf.nn.relu, name='fc6')(feature)
    hidden = keras.layers.Dense(dim, activation=tf.nn.relu, name='fc7')(hidden)
    return hidden


def fastrcnn_Xconv1fc_head(feature, num_convs, norm=None):
    """
    Args:
        feature (NHWC):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers
        norm (str or None): either None or 'GN'
    Returns:
        2D head feature
    """
    l = feature
    for k in range(num_convs):
        l = keras.layers.Conv2D(_C.FPN.FRCNN_CONV_HEAD_DIM, (3, 3), activation=tf.nn.relu, padding="same",
                                name="conv{}".format(k))(l)
        if norm is not None:
            l = group_normalization(l)
    l = keras.layers.Dense(_C.FPN.FRCNN_FC_HEAD_DIM, activation=tf.nn.relu)(l)
    return l


def fastrcnn_4conv1fc_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, **kwargs)


def fastrcnn_4conv1fc_gn_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, norm='GN', **kwargs)


class FastRCNNHead(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """
    def __init__(self, proposals, box_logits, label_logits, gt_boxes, bbox_regression_weights):
        """
        Args:
            proposals: BoxProposals
            box_logits: Nx#classx4 or Nx1x4, the output of the head
            label_logits: Nx#class, the output of the head
            gt_boxes: Mx4
            bbox_regression_weights: a 4 element tensor
        """
        self.proposals = proposals
        self.box_logits = box_logits
        self.label_logits = label_logits
        self.gt_boxes= gt_boxes
        self.bbox_regression_weights = bbox_regression_weights
        self._bbox_class_agnostic = int(box_logits.shape[1]) == 1
        self._num_classes = tf.shape(box_logits)[1]
        self.is_training = True

    def fg_box_logits(self):
        """ Returns: #fg x ? x 4 """
        return tf.gather(self.box_logits, self.proposals.fg_inds(), name='fg_box_logits')

    def losses(self):
        encoded_fg_gt_boxes = encode_bbox_target(
            tf.gather(self.gt_boxes, self.proposals.fg_inds_wrt_gt),
            self.proposals.fg_boxes()) * self.bbox_regression_weights
        return fastrcnn_losses(
            self.proposals.labels, self.label_logits,
            encoded_fg_gt_boxes, self.fg_box_logits()
        )

    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.proposals.boxes, 1),
                          [1, self._num_classes, 1])   # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.box_logits / self.bbox_regression_weights,
            anchors
        )
        return decoded_boxes

    def decoded_output_boxes_for_true_label(self):
        """ Returns: Nx4 decoded boxes """
        return self._decoded_output_boxes_for_label(self.proposals.labels)

    def decoded_output_boxes_for_predicted_label(self):
        """ Returns: Nx4 decoded boxes """
        return self._decoded_output_boxes_for_label(self.predicted_labels())

    def _decoded_output_boxes_for_label(self, labels):
        assert not self._bbox_class_agnostic
        indices = tf.stack([
            tf.range(tf.size(labels, out_type=tf.int64)),
            labels
        ])
        needed_logits = tf.gather_nd(self.box_logits, indices)
        decoded = decode_bbox_target(
            needed_logits / self.bbox_regression_weights,
            self.proposals.boxes
        )
        return decoded

    def decoded_output_boxes_class_agnostic(self):
        """ Returns: Nx4 """
        assert self._bbox_class_agnostic
        box_logits = tf.reshape(self.box_logits, [-1, 4])
        decoded = decode_bbox_target(
            box_logits / self.bbox_regression_weights,
            self.proposals.boxes
        )
        return decoded

    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

    def predicted_labels(self):
        """ Returns: N ints """
        return tf.argmax(self.label_logits, axis=1, name='predicted_labels')

