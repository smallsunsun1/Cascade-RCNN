import tensorflow as tf

from .data import tf_get_multilevel_rpn_anchor_input, tf_get_rpn_anchor_input


def transform_img_and_boxes(image, boxes, target_size, training=True):
    target_h = target_size[0]
    target_h_float = tf.cast(target_h, tf.float32)
    target_w = target_size[1]
    target_w_float = tf.cast(target_w, tf.float32)
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
    image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, axis=0), [new_h, new_w]), axis=0)
    image = tf.pad(image, [[pad_h_top, pad_h_bottom], [pad_w_left, pad_w_right], [0, 0]])
    box_l = (boxes[:, 0] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w_float
    box_r = (boxes[:, 2] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w_float
    box_t = (boxes[:, 1] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h_float
    box_b = (boxes[:, 3] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h_float
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
    box_l = box_l * target_w_float
    box_r = box_r * target_w_float
    box_t = box_t * target_h_float
    box_b = box_b * target_w_float
    boxes = tf.stack([box_l, box_t, box_r, box_b], axis=1)
    return image, boxes


def tf_transform(data, training=True):
    file_data = tf.io.read_file(data["filename"])
    data["boxes"] = tf.reshape(tf.io.decode_raw(data["boxes"], tf.float32), shape=[-1, 4])
    image = tf.image.decode_jpeg(file_data, 3)
    shape2d = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = shape2d[1] / shape2d[0]
    # new_height = tf.random.uniform([], 600, 800, tf.int32)
    # new_width = tf.minimum(tf.cast(tf.cast(new_height, tf.float32) * scale, tf.int32), 1333)
    # new_height = tf.random.uniform([], 600, 1333, tf.int32) // 32 * 32
    # new_width = tf.minimum(tf.cast(tf.cast(new_height, tf.float32) * scale, tf.int32), 1333) // 32 * 32
    new_height = 720
    new_width = 720
    target_size = [new_height, new_width]
    data["image"], data["boxes"] = transform_img_and_boxes(image, data["boxes"], target_size, training)
    data["class"] = tf.reshape(tf.decode_raw(data['class'], tf.int32), shape=[-1]) + 1
    data["is_crowd"] = tf.reshape(tf.decode_raw(data['is_crowd'], tf.int32), shape=[-1])
    data["image"] = tf.expand_dims(data["image"], axis=0)
    return data


def preprocess(data, fpn_mode=False):
    if fpn_mode:
        multilevel_anchor_inputs = tf_get_multilevel_rpn_anchor_input(data['image'], data['boxes'],
                                                                      data['is_crowd'])
        for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):
            data['anchor_labels_lvl{}'.format(i + 2)] = anchor_labels
            data['anchor_boxes_lvl{}'.format(i + 2)] = anchor_boxes
    else:
        data["anchor_labels"], data['anchor_boxes'] = tf_get_rpn_anchor_input(data['image'], data['boxes'],
                                                                              data['is_crowd'])
    data['boxes'] = tf.gather_nd(data['boxes'], tf.where(tf.equal(data['is_crowd'], 0)))
    data['gt_labels'] = tf.gather_nd(data['class'], tf.where(tf.equal(data['is_crowd'], 0)))
    return data


def input_fn(filenames, training=True):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(x, features={"filename": tf.io.FixedLenFeature([], tf.string),
                                                          "boxes": tf.io.FixedLenFeature([], tf.string),
                                                          "class": tf.io.FixedLenFeature([], tf.string),
                                                          "is_crowd": tf.io.FixedLenFeature([], tf.string)}),
        num_parallel_calls=10)
    if training:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(50000))
    dataset = dataset.map(lambda x: tf_transform(x, training), 10)
    dataset = dataset.map(lambda x: preprocess(x, False), 10)
    dataset = dataset.prefetch(-1)
    return dataset


def read_img(filename, target_height, target_width):
    content = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(content, 3)
    image = tf.image.resize_image_with_pad(image, target_height, target_width)
    features = {}
    features['image'] = tf.expand_dims(image, 0)
    return features


def test_input_fn(filenames, target_height, target_width):
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(lambda x:read_img(x, target_height, target_width), num_parallel_calls=10)
    dataset = dataset.prefetch(-1)
    return dataset





