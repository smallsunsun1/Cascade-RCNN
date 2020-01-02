import tensorflow as tf
import sys

sys.path.append("..")
from config.config import _C
from .model_box import clip_boxes
from .model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs
from util.box_ops import tf_iou


class CascadeRCNNHead(object):
    def __init__(self, proposals, roi_func, fastrcnn_head_func, gt_targets, image_shape2d, num_classes, is_train):
        """
        Args:
            proposals: BoxProposals
            roi_func (boxes -> features): a function to crop features with rois
            fastrcnn_head_func (features -> features): the fastrcnn head to apply on the cropped features
            gt_targets (gt_boxes, gt_labels):
        """
        self.proposals = proposals
        self.roi_func = roi_func
        self.fastrcnn_head_func = fastrcnn_head_func
        self.gt_boxes, self.gt_labels = gt_targets
        self.image_shape2d = image_shape2d
        self.num_classes = num_classes
        self.num_cascade_stages = len(_C.CASCADE.IOUS)
        self.is_train = is_train

        @tf.custom_gradient
        def scale_gradient(x):
            return x, lambda dy: dy * (1.0 / self.num_cascade_stages)
        self.scale_gradient = scale_gradient
        ious = _C.CASCADE.IOUS
        assert self.num_cascade_stages == 3, "Only 3-stage cascade was implemented!"
        H1, B1 = self.run_head(self.proposals, 0)
        B1_proposal = self.match_box_with_gt(B1, ious[1])
        H2, B2 = self.run_head(B1_proposal, 1)
        B2_proposals = self.match_box_with_gt(B2, ious[2])
        H3, B3 = self.run_head(B2_proposals, 2)
        self._cascade_boxes = [B1, B2, B3]
        self._heads = [H1, H2, H3]


    def run_head(self, proposals, stage):
        """

        :param proposals: BoxProposals
        :param stage: 0, 1, 2
        :return: FastRCNNHead
                 Nx4, updated boxes
        """
        reg_weights = tf.convert_to_tensor(_C.CASCADE.BBOX_REG_WEIGHTS[stage], dtype=tf.float32)
        pooled_feature = self.roi_func(proposals.boxes)
        pooled_feature = self.scale_gradient(pooled_feature)
        head_feature = self.fastrcnn_head_func(pooled_feature)
        label_logits, box_logits = fastrcnn_outputs(head_feature, self.num_classes, class_agnostic_regression=True)
        head = FastRCNNHead(proposals, box_logits, label_logits, self.gt_boxes, reg_weights)
        refined_boxes = head.decoded_output_boxes_class_agnostic()
        refined_boxes = clip_boxes(refined_boxes, self.image_shape2d)
        return head, tf.stop_gradient(refined_boxes)

    def match_box_with_gt(self, boxes, iou_threshold):
        """

        :param boxes: Nx4
        :param iou_threshold:
        :return: BoxProposals
        """
        if self.is_train:
            iou = tf_iou(boxes, self.gt_boxes)  # NxM
            max_iou_per_box = tf.reduce_max(iou, axis=1)  # N
            best_iou_ind = tf.argmax(iou, axis=1) # N
            labels_per_box = tf.gather(self.gt_labels, best_iou_ind)
            fg_mask = tf.greater_equal(max_iou_per_box, iou_threshold)
            fg_inds_wrt_gt = tf.boolean_mask(best_iou_ind, fg_mask)
            labels_per_box = tf.stop_gradient(labels_per_box * tf.cast(fg_mask, tf.int32))
            return BoxProposals(boxes, labels_per_box, fg_inds_wrt_gt)
        else:
            return BoxProposals(boxes)

    def losses(self):
        ret = []
        for idx, head in enumerate(self._heads):
            ret.extend(head.losses())
        return ret

    def decoded_output_boxes(self):
        """

        :return: Nxclassx4
        """
        ret = self._cascade_boxes[-1]
        ret = tf.expand_dims(ret, axis=1)
        return tf.tile(ret, [1, self.num_classes, 1])

    def output_scores(self):
        scores = [head.output_scores('cascade_scores_stage{}'.format(idx + 1)) for idx, head in enumerate(self._heads)]
        return tf.multiply(tf.add_n(scores), (1.0 / self.num_cascade_stages))
