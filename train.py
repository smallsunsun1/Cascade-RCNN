import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
import re

from tensorflow import keras
from tensorflow.contrib import distribute

from config.config import _C
from model import model_frcnn
from model.basemodel import resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
from model.model_rpn import rpn_head, generate_rpn_proposals, rpn_losses, RPNHead
from model.model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses, \
    slice_feature_and_anchors
from model.model_box import RPNAnchors, clip_boxes, roi_align, decoded_output_boxes
from model.model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions, \
    sample_fast_rcnn_targets, fastrcnn_predictions_v2
from model.model_cascade import CascadeRCNNHead
from util.common import image_preprocess
from util.data import tf_get_all_anchors, tf_get_all_anchors_fpn
from util.data_loader import input_fn, test_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


def map_boxes_back(boxes, features):
    h_pre = features['h_pre']
    w_pre = features['w_pre']
    h_now = features['h_now']
    w_now = features['w_now']
    scale = features['scale']  
    scale_now = w_now / h_now
    if scale > 1:
        true_h = w_now / scale
        pad_h_top = (h_now - true_h) // 2
        pad_h_bottom = h_now - true_h - pad_h_top
        pad_w_left = 0
        pad_w_right = 0
        true_w = w_now
    else:
        true_w = h_now * scale
        pad_w_left = (w_now - true_w) // 2
        pad_w_right = w_now - true_w - pad_w_left
        pad_h_top = 0
        pad_h_bottom = 0
        true_h = h_now
    boxes[:, 0] = boxes[:, 0] - pad_w_left
    boxes[:, 1] = boxes[:, 1] - pad_h_top
    boxes[:, 2] = boxes[:, 2] - pad_w_left
    boxes[:, 3] = boxes[:, 3] - pad_h_top
    boxes[:, 0] = boxes[:, 0] / true_w * w_pre
    boxes[:, 1] = boxes[:, 1] / true_h * h_pre
    boxes[:, 2] = boxes[:, 2] / true_w * w_pre
    boxes[:, 3] = boxes[:, 3] / true_h * h_pre
    return boxes 


def resnet_c4_model_fn(features, labels, mode, params):
    """parameter defination part"""
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

    """model definition part"""
    image = image_preprocess(features['image'])
    featuremap = resnet_c4_backbone(image, resnet_num_blocks[:3], is_train)
    image_shape2d = tf.shape(image)[1:3]
    rpn_label_logits, rpn_box_logits = rpn_head(featuremap, head_dim, num_anchors)
    if mode != tf.estimator.ModeKeys.PREDICT:
        anchors = RPNAnchors(tf_get_all_anchors(), features['anchor_labels'], features['anchor_boxes'])
    else:
        anchors = RPNAnchors(tf_get_all_anchors(), None, None)
    anchors = anchors.narrow_to(featuremap)
    pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # x1y1x2y2
    proposals, proposal_scores = generate_rpn_proposals(
        tf.reshape(pred_boxes_decoded, [-1, 4]),
        tf.reshape(rpn_label_logits, [-1]),
        image_shape2d,
        _C.RPN.TRAIN_PRE_NMS_TOPK if mode == tf.estimator.ModeKeys.TRAIN else _C.RPN.TEST_PRE_NMS_TOPK,
        _C.RPN.TRAIN_POST_NMS_TOPK if mode == tf.estimator.ModeKeys.TRAIN else _C.RPN.TEST_POST_NMS_TOPK)
    # rpn_size = tf.shape(proposals)[0]
    # rpn_boxes = tf.gather(proposals, tf.where(tf.greater(proposals, 0.5)))

    proposals = BoxProposals(proposals)
    if mode != tf.estimator.ModeKeys.PREDICT:
        rpn_loss = rpn_losses(anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits)
        # targets = [features[k] for k in ['boxes', 'gt_labels', 'gt_masks'] if k in features.keys()]
        # gt_boxes, gt_labels, *_ = targets
        gt_boxes = features['boxes']
        gt_labels = features['gt_labels']
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
    final_boxes, final_scores, final_labels, valid_detections = fastrcnn_predictions_v2(decoded_boxes, label_scores)
    # final_boxes, final_scores, final_labels = fastrcnn_predictions(decoded_boxes, label_scores)
    global_step = tf.train.get_or_create_global_step()
    if mode != tf.estimator.ModeKeys.PREDICT:
        trainable_weights = tf.trainable_variables()
        weight_loss = 0.0
        for i, ele in enumerate(trainable_weights):
            if re.search('.*/kernel', ele.name):
                weight_loss += tf.reduce_sum(tf.square(ele) * weight_decay)
        total_cost = tf.add_n(rpn_loss + all_loss, 'total_cost')
        tf.summary.scalar('total_cost', total_cost)
        if is_train:
            learning_rate = tf.train.piecewise_constant(global_step, lr_schedule,
                                                        values=[tf.convert_to_tensor(0.01 * 0.33, tf.float32)] + [
                                                            learning_rate * (0.1 ** i) for i in
                                                            range(len(lr_schedule))])
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            # opt = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(total_cost, global_step)
            return tf.estimator.EstimatorSpec(mode, loss=total_cost, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=total_cost)
    else:
        predictions = {'boxes': final_boxes[0, :valid_detections[0]],
                       'labels': final_labels[0, :valid_detections[0]],
                       'scores': final_scores[0, :valid_detections[0]],
                       'image': features['image'],
                       # 'rpn_boxes': rpn_boxes,
                       # 'rpn_size': rpn_size,
                       'valid_detection': valid_detections}
        return tf.estimator.EstimatorSpec(mode, predictions)


def resnet_fpn_model_fn(features, labels, mode, params):
    """parameter definition part"""
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

    """model definition part"""
    image = image_preprocess(features["image"])
    c2345 = resnet_fpn_backbone(image, resnet_num_blocks, is_train)
    p23456 = fpn_model(c2345)
    image_shape2d = tf.shape(image)[1:3]
    all_anchors_fpn = tf_get_all_anchors_fpn()
    model_rpn_head = RPNHead(_C.FPN.NUM_CHANNEL, len(_C.RPN.ANCHOR_RATIOS))
    rpn_outputs = [model_rpn_head(pi) for pi in p23456]
    multilevel_label_logits = [k[0] for k in rpn_outputs]
    multilevel_box_logits = [k[1] for k in rpn_outputs]
    #debug_op = tf.print({"debug_inf": tf.convert_to_tensor("now in here")})
    #with tf.control_dependencies([debug_op]):
    #    image_shape2d = tf.identity(image_shape2d)
    if mode != tf.estimator.ModeKeys.PREDICT:
        multilevel_anchors = [RPNAnchors(all_anchors_fpn[i], features['anchor_labels_lvl{}'.format(i + 2)],
                                     features['anchor_boxes_lvl{}'.format(i+2)]) for i in range(len(all_anchors_fpn))]
    else:
        multilevel_anchors = [RPNAnchors(all_anchors_fpn[i], None,
                                     None) for i in range(len(all_anchors_fpn))]
    slice_feature_and_anchors(p23456, multilevel_anchors)
    # Multi-Level RPN Proposals
    multilevel_pred_boxes = [anchor.decode_logits(logits) for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)]
    proposal_boxes, proposal_scores = generate_fpn_proposals(multilevel_pred_boxes, multilevel_label_logits, image_shape2d, is_train)
    proposals = BoxProposals(proposal_boxes)
    gt_boxes = None
    gt_labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        losses = multilevel_rpn_losses(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
        gt_boxes = features['boxes']
        gt_labels = features['gt_labels']
        proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)
    fastrcnn_head_func = getattr(model_frcnn, _C.FPN.FRCNN_HEAD_FUNC)
    if not _C.FPN.CASCADE:
        roi_feature_fastrcnn = multilevel_roi_align(p23456[:4], proposals.boxes, 7)
        head_feature = fastrcnn_head_func(roi_feature_fastrcnn)
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(head_feature, num_classes)
        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                         gt_boxes, tf.convert_to_tensor(bbox_reg_weights, tf.float32))
    else:
        def roi_func(boxes):
            return multilevel_roi_align(p23456[:4], boxes, 7)
        fastrcnn_head = CascadeRCNNHead(proposals, roi_func, fastrcnn_head_func,
                                            (gt_boxes, gt_labels), image_shape2d, num_classes, mode != tf.estimator.ModeKeys.PREDICT)
    decoded_boxes = fastrcnn_head.decoded_output_boxes()
    decoded_boxes = clip_boxes(decoded_boxes, image_shape2d)
    label_scores = fastrcnn_head.output_scores()
    final_boxes, final_scores, final_labels = fastrcnn_predictions(decoded_boxes, label_scores)
    #final_boxes, final_scores, final_labels, valid_detections = fastrcnn_predictions_v2(decoded_boxes, label_scores)
    global_step = tf.train.get_or_create_global_step()
    if mode != tf.estimator.ModeKeys.PREDICT:
        all_losses = fastrcnn_head.losses()
        trainable_weights = tf.trainable_variables()
        weight_loss = 0.0
        for i, ele in enumerate(trainable_weights):
            if re.search('.*/kernel', ele.name):
                weight_loss += tf.reduce_sum(tf.square(ele) * weight_decay)

        #print_op = tf.print({'rpn_loss': tf.add_n(losses),
        #                     'frcnn_loss': tf.add_n(all_losses)})
        #with tf.control_dependencies([print_op]):
        total_cost = tf.add_n(losses + all_losses, "total_cost") 
        total_cost = tf.add(total_cost, weight_loss, 'all_total_cost')
        if is_train:
            learning_rate = tf.train.piecewise_constant(global_step, lr_schedule,
                                                        values=[tf.convert_to_tensor(0.01 * 0.33, tf.float32)] + [
                                                            learning_rate * (0.1 ** i) for i in
                                                            range(len(lr_schedule))])
            tf.summary.scalar("learning_rate", learning_rate)
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            # opt = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(total_cost, global_step)
            return tf.estimator.EstimatorSpec(mode, loss=total_cost, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=total_cost)
    else:
        #predictions = {'boxes': final_boxes[0, :valid_detections[0]],
        #               'labels': final_labels[0, :valid_detections[0]],
        #               'scores': final_scores[0, :valid_detections[0]],
        #               'image': features['image'],
        #               'valid_detection': valid_detections}
        predictions = {'boxes': final_boxes,
                       'labels': final_labels,
                       'scores': final_scores,
                       'image': features['image'],
                       'original_image': features['original_image'],
                       'h_pre': features['h_pre'],
                       'w_pre': features['w_pre'],
                       'h_now': features['h_now'],
                       'w_now': features['w_now'],
                       'scale': features['scale']
                      }
        return tf.estimator.EstimatorSpec(mode, predictions)

model_dict = {"rpn": resnet_c4_model_fn,
              "fpn": resnet_fpn_model_fn}


if __name__ == "__main__":
    # tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='rpn', help='training_model')
    parser.add_argument('--model_dir', default='./rpn_model_v2', help='where to store the model')
    parser.add_argument("--train_filename", default="/home/admin-seu/sss/master_work/data/train.record",
                        help='train filename')
    parser.add_argument("--eval_filename", default="/home/admin-seu/sss/master_work/data/eval.record",
                        help='eval filename')
    parser.add_argument("--test_filename", default="/home/admin-seu/sss/yolo-V3/data/test.txt", help="test_filename")
    parser.add_argument("--gpus", default=2, help='num_of_gpus', type=int)
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
    #dataset = input_fn(args.train_filename, True, _C.MODE_FPN)
    #for idx, element in enumerate(dataset):
    #    print(idx)
    #    print(element)
    #    if idx == 10:
    #        break


    if args.gpus > 0:
        strategy = distribute.MirroredStrategy(num_gpus=args.gpus)
        session_configs = tf.ConfigProto(allow_soft_placement=True)
        session_configs.gpu_options.allow_growth = True
        config = tf.estimator.RunConfig(train_distribute=strategy, session_config=session_configs,
                                        log_step_count_steps=100, save_checkpoints_steps=10000,
                                        eval_distribute=strategy, save_summary_steps=500)
        estimator = tf.estimator.Estimator(model_dict[args.model], args.model_dir, config,
                                           params)
    else:
        config = tf.estimator.RunConfig(save_checkpoints_steps=10000, save_summary_steps=500, log_step_count_steps=100)
        estimator = tf.estimator.Estimator(model_dict[args.model], args.model_fir, config,
                                           params)
    train_spec = tf.estimator.TrainSpec(lambda: input_fn(args.train_filename, True, _C.MODE_FPN), max_steps=None)
    eval_spec = tf.estimator.EvalSpec(lambda: input_fn(args.eval_filename, False, _C.MODE_FPN), steps=1000)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    res = estimator.predict(lambda: test_input_fn(args.test_filename, 960, 960), yield_single_examples=False)
    # res = estimator.predict(lambda :input_fn(args.eval_filename, False), yield_single_examples=False)
    score_thresh = 0.5
    for idx, ele in enumerate(res):
        image = ele["original_image"].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print("current image index: ", idx)
        print("boxes: ", ele["boxes"])
        print("labels: ", ele["labels"])
        print("scores: ", ele["scores"])
        ele["boxes"] = map_boxes_back(ele["boxes"], ele)
        for num_idx, box in enumerate(ele["boxes"]):
            if ele["scores"][num_idx] < score_thresh:
                continue
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(image, '{}'.format(ele["labels"][num_idx]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
        cv2.imwrite("./detect_result/{}.jpg".format(idx), image)
        #print("boxes: ", ele["boxes"])
        #print("labels: ", ele["labels"])
        #print("scores: ", ele["scores"])
        # print("rpn_boxes: ", ele["rpn_boxes"])
        # print("rpn_size: ", ele["rpn_size"])
        #print('valid_detection: ', ele["valid_detection"])
        if idx == 100:
            break

