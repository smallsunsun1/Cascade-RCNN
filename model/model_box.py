import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")

from tensorflow import keras

from config.config import _C

def clip_boxes(boxes, windows):
    """

    :param boxes: nx3, xyxy
    :param windows:
    :return:
    """
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(windows, axis=[0]), multiples=[2])
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32))
    return boxes

def encode_bbox_target(boxes, anchors):
    """

    :param boxes: (..., 4) float32
    :param anchors: (..., 4) float32
    :return:    (..., 4), float32 with the same shape
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, [-1, 2, 2])
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x1y1 + anchors_x2y2) * 0.5
    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5
    # Note that here not all boxes are valid. some may be zero
    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha) # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=-1) # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))

def decode_bbox_target(box_predictions, anchors):
    """
        Args:
            box_predictions: (..., 4), logits
            anchors: (..., 4), floatbox. Must have the same shape
        Returns:
            box_decoded: (..., 4), float32. With the same shape.
        """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5
    clip = np.log(_C.PREPROC.MAX_SIZE / 16.)
    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)

def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
        Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
        Args:
            image: NHWC
            boxes: nx4, x1y1x2y2
            box_ind: (n,)
            crop_size (int):
        Returns:
            n,C,size,size
        """
    assert isinstance(crop_size, int), crop_size
    if pad_border:
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
        boxes = boxes + 1
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
            The way tf.image.crop_and_resize works (with normalized box):
            Initial point (the value of output[0]): x0_box * (W_img - 1)
            Spacing: w_box * (W_img - 1) / (W_crop - 1)
            Use the above grid to bilinear sample.
            However, what we want is (with fpcoor box):
            Spacing: w_box / W_crop
            Initial point: x0_box + spacing/2 - 0.5
            (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
             (0.0, 0.0) is the same as pixel value (0, 0))
            This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
            Returns:
                y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)
        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)
        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]
        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]
        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[1:3]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    return ret

def decoded_output_boxes(proposals, num_classes, box_logits, bbox_regression_weights):
    """ Returns: N x #class x 4 """
    anchors = tf.tile(tf.expand_dims(proposals.boxes, 1),
                      [1, num_classes, 1])   # N x #class x 4
    decoded_boxes = decode_bbox_target(
        box_logits / bbox_regression_weights,
        anchors
    )
    return decoded_boxes

def roi_align(feature_map, boxes, resolution):
    """

    :param feature_map: 1xHxWxC
    :param boxes: Nx4 flaotbox
    :param resolution: output spatial resolution
    :return: NxresxresxC
    """
    ret = crop_and_resize(feature_map, boxes, tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
                          resolution * 2)
    ret = keras.layers.AveragePooling2D((2, 2), 2, padding="same")(ret)
    return ret

class RPNAnchors():
    """
        boxes (FS x FS x NA x 4): The anchor boxes.
        gt_labels (FS x FS x NA):
        gt_boxes (FS x FS x NA x 4): Groundtruth boxes corresponding to each anchor.
    """
    def __init__(self, boxes, gt_labels, gt_boxes):
        self.boxes = boxes
        self.gt_labels = gt_labels
        self.gt_boxes = gt_boxes

    def encoded_gt_boxes(self):
        return encode_bbox_target(self.gt_boxes, self.boxes)

    def decode_logits(self, logits):
        return decode_bbox_target(logits, self.boxes)

    def narrow_to(self, featuremap):
        shape2d = tf.shape(featuremap)[1:3]
        slice3d = tf.concat([shape2d, [-1]], axis=0)
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)
        boxes = tf.slice(self.boxes, [0, 0, 0, 0], slice4d)
        if self.gt_labels is not None:
            gt_labels = tf.slice(self.gt_labels, [0, 0, 0], slice3d)
        else:
            gt_labels = self.gt_labels
        if self.gt_boxes is not None:
            gt_boxes = tf.slice(self.gt_boxes, [0, 0, 0, 0], slice4d)
        else:
            gt_boxes = self.gt_boxes
        return RPNAnchors(boxes, gt_labels, gt_boxes)



