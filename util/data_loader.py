import tensorflow as tf
import numpy as np


def transform_img_and_boxes(image, boxes, target_size, training=True):
    target_h = target_size[0]
    target_w = target_size[1]
    image_shape = tf.shape(image)
    img_h = image_shape[0]
    img_w = image_shape[1]
    h_scale = tf.cast(tf.divide(target_h, img_h), tf.float32)
    w_scale = tf.cast(tf.divide(target_w, img_w), tf.float32)
    scale = tf.minimum(h_scale, w_scale)
    new_h = tf.cast(tf.multiply(tf.cast(img_h, tf.float32), scale), tf.int32)
    new_w = tf.cast(tf.multiply(tf.cast(img_w, tf.float32), scale), tf.int32)
    pad_h_top = (target_h - new_h) // 2
    pad_h_bottom = target_h - new_h - pad_h_top
    pad_w_left = (target_w - new_w) // 2
    pad_w_right = target_w - new_w - pad_w_left
    image = tf.squeeze(tf.image.resize(tf.expand_dims(image, axis=0), [new_h, new_w]), axis=0)
    image = tf.pad(image, [[pad_h_top, pad_h_bottom], [pad_w_left, pad_w_right], [0, 0]])
    box_l = (boxes[:, 0] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w
    box_r = (boxes[:, 2] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w
    box_t = (boxes[:, 1] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h
    box_b = (boxes[:, 3] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h
    if training:
        p1 = tf.random.uniform([], 0, 10)
        p2 = tf.random.uniform([], 0, 10)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.1, 0.2)
        image = tf.image.random_hue(image, 0.1)
        image = tf.clip_by_value(image, 0, 255)
        cond1 = tf.greater(p1, 5.0)
        cond2 = tf.greater(p2, 5.0)

        def flip_left_right(image, box_l, box_r, box_t, box_b):
            image = tf.image.flip_left_right(image)
            box_l = 1.0 - box_l
            box_r = 1.0 - box_r
            return image, box_r, box_l, box_t, box_b

        def flip_top_down(image, box_l, box_r, box_t, box_b):
            image = tf.image.flip_up_down(image)
            box_t = 1.0 - box_t
            box_b = 1.0 - box_b
            return image, box_l, box_r, box_b, box_t

        image, box_l, box_r, box_t, box_b = tf.cond(cond1, lambda: flip_left_right(image, box_l, box_r, box_t, box_b),
                                                    lambda: (image, box_l, box_r, box_t, box_b))
        image, box_l, box_r, box_t, box_b = tf.cond(cond2, lambda: flip_top_down(image, box_l, box_r, box_t, box_b),
                                                    lambda: (image, box_l, box_r, box_t, box_b))
    boxes = tf.stack([box_l, box_t, box_r, box_b, boxes[:, 4]], axis=1)
    return image, boxes


def tf_transform(data, training=True):
    file_data = tf.io.read_file(data["filename"])
    data["boxes"] = tf.reshape(tf.io.decode_raw(data["boxes"], tf.float32), shape=[-1, 5])
    image = tf.image.decode_jpeg((file_data, 3))
    shape2d = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = shape2d[1] / shape2d[0]
    new_height = tf.random.uniform([], 600, 800, tf.int32)
    new_width = tf.minimum(tf.cast(tf.cast(new_height, tf.float32) * scale, tf.int32), 1333)
    target_size = [new_height, new_width]
    data["image"], data["boxes"] = transform_img_and_boxes(image, data["boxes"], target_size, training)
    return data


def input_fn(filenames, training=True):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(x, features={"filename": tf.io.FixedLenFeature([], tf.string),
                                                          "boxes": tf.io.FixedLenFeature([], tf.string)}))
    dataset = dataset.map(lambda x: tf_transform(x, training))
    return dataset


