import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
import re

from tensorflow import keras
from tensorflow.contrib import distribute

from config.config import _C
from model.basemodel import resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
from model.model_rpn import rpn_head, generate_rpn_proposals, rpn_losses
from model.model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses, \
    slice_feature_and_anchors
from model.model_box import RPNAnchors, clip_boxes, roi_align, decoded_output_boxes
from model.model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions, sample_fast_rcnn_targets
from util.common import image_preprocess
from util.data import tf_get_all_anchors, tf_get_all_anchors_fpn
from util.data_loader import input_fn

tf.logging.set_verbosity(tf.logging.INFO)


def resnet_c4_model_fn(features, labels, mode, params):
    """参数定义部分"""
    is_train = (mode == tf.estimator.ModeKeys.TRAIN)
    resnet_num_blocks = params["RESNET_NUM_BLOCKS"]
    num_anchors = params["num_anchors"]
    head_dim = params["head_dim"]
    resolution = params["resolution"]
    num_classes = params["num_classes"]
    bbox_reg_weights = params["bbox_reg_weights"]
    weight_decay = params["weight_decay"]
    learning_rate = params["learning_rate"]
    lr_schedule = params["lr_schedule"]


    """模型定义部分"""
    image = image_preprocess(features['image'])
    featuremap = resnet_c4_backbone(image, resnet_num_blocks[:3], is_train)
    image_shape2d = tf.shape(image)[1:3]
    rpn_label_logits, rpn_box_logits = rpn_head(featuremap, head_dim, num_anchors)
    if is_train:
        anchors = RPNAnchors(tf_get_all_anchors(), features['anchor_labels'], features['anchor_boxes'])
    else:
        anchors = RPNAnchors(tf_get_all_anchors(), None, None)
    anchors = anchors.narrow_to(featuremap)
    pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  #x1y1x2y2
    proposals, proposal_scores = generate_rpn_proposals(
        tf.reshape(pred_boxes_decoded, [-1, 4]),
        tf.reshape(rpn_label_logits, [-1]),
        image_shape2d,
        _C.RPN.TRAIN_PRE_NMS_TOPK if mode == tf.estimator.ModeKeys.TRAIN else _C.RPN.TEST_PRE_NMS_TOPK,
        _C.RPN.TRAIN_POST_NMS_TOPK if mode == tf.estimator.ModeKeys.TRAIN else _C.RPN.TEST_POST_NMS_TOPK)
    proposals = BoxProposals(proposals)
    if mode != tf.estimator.ModeKeys.PREDICT:
        rpn_loss = rpn_losses(anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits)
        targets = [features[k] for k in ['boxes', 'gt_labels', 'gt_masks'] if k in features.keys()]
        gt_boxes, gt_labels, *_ = targets
        proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)
    boxes_on_featuremap = proposals.boxes * (1.0 / _C.RPN.ANCHOR_STRIDE)
    roi_resized = roi_align(featuremap, boxes_on_featuremap, resolution)
    feature_fastrcnn = resnet_conv5(roi_resized, resnet_num_blocks[-1], is_train)
    # Keep C5 feature to be shared with mask branch
    feature_gap = keras.layers.GlobalAveragePooling2D()(feature_fastrcnn)
    fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(feature_gap, num_classes)
    bbox_reg_weights_tensor = tf.convert_to_tensor(bbox_reg_weights, tf.float32)
    if mode != tf.estimator.ModeKeys.PREDICT:
        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                     gt_boxes, bbox_reg_weights_tensor)
        all_loss = fastrcnn_head.losses()
    label_scores = tf.nn.softmax(fastrcnn_label_logits)
    decoded_boxes = decoded_output_boxes(proposals, num_classes, fastrcnn_box_logits, bbox_reg_weights_tensor)
    decoded_boxes = clip_boxes(decoded_boxes, image_shape2d)
    final_boxes, final_scores, final_labels = fastrcnn_predictions(decoded_boxes, label_scores)
    global_step = tf.train.get_or_create_global_step()
    if mode != tf.estimator.ModeKeys.PREDICT:
        trainable_weights = tf.trainable_variables()
        weight_loss = 0.0
        for i, ele in enumerate(trainable_weights):
            if re.search('.*/kernel', ele.name):
                weight_loss += tf.reduce_sum(tf.square(ele) * weight_decay)
        total_cost = tf.add_n(rpn_loss + all_loss + [weight_loss], 'total_cost')
        tf.summary.scalar('total_cost', total_cost)
        if is_train:
            learning_rate = tf.train.piecewise_constant(global_step, lr_schedule,
                                                        values=[learning_rate * i for i in range(len(lr_schedule) + 1)])
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(total_cost, global_step)
            return tf.estimator.EstimatorSpec(mode, loss=total_cost, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=total_cost)
    else:
        predictions = {'boxes': final_boxes,
                       'labels': final_labels,
                       'scores': final_scores}
        return tf.estimator.EstimatorSpec(mode, predictions)








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='rpn', help='training_model')
    parser.add_argument('--model_dir', default='./rpn_model', help='where to store the model')
    parser.add_argument("--train_filename", default="./train.txt", help='train filename')
    parser.add_argument("--eval_filename", default="./eval.txt", help='eval filename')
    parser.add_argument("--gpus", default=1, help='num_of_gpus')
    args = parser.parse_args()

    params = {}
    params["RESNET_NUM_BLOCKS"] = _C.BACKBONE.RESNET_NUM_BLOCKS
    params["num_anchors"] = _C.RPN.NUM_ANCHOR
    params["head_dim"] = _C.RPN.HEAD_DIM
    params["resolution"] = 14
    params["num_classes"] = _C.DATA.NUM_CLASS
    params["bbox_reg_weights"] = _C.FRCNN.BBOX_REG_WEIGHTS
    params["weight_decay"] = _C.TRAIN.WEIGHT_DECAY
    params["learning_rate"] = _C.TRAIN.BASE_LR
    params["lr_schedule"] = [_C.TRAIN.WARMUP] + _C.TRAIN.LR_SCHEDULE

    if args.gpus > 0:
        strategy = distribute.MirroredStrategy(num_gpus=args.gpus)
        session_configs = tf.ConfigProto(allow_soft_placement=True)
        session_configs.gpu_options.allow_growth = True
        config = tf.estimator.RunConfig(train_distribute=strategy, session_config=session_configs,
                                        log_step_count_steps=100, save_checkpoints_steps=5000,
                                        eval_distribute=strategy, save_summary_steps=500)
        estimator = tf.estimator.Estimator(resnet_c4_model_fn, args.model_dir, config,
                                           params)




