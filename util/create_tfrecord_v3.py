import argparse
import io
import os
import hashlib
import logging
import random
import json

import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

parser = argparse.ArgumentParser()
parser.add_argument('--list_path', default='', help='Path to data list')
parser.add_argument('--output_path', default='', help='Path to output tfrecord')
parser.add_argument('--label_map_path', default='/mnt/WXRG0243/jhsun/Gitlab/camp_workspace/car_detection/S3/label_map.pbtxt', help='Path to label map')
parser.add_argument('--shuffle', default=True, type=bool, help='Whether to shuffle the data')
FLAGS = parser.parse_args()

pics = 0
gts = 0
simple = 0
hard = 0

def transform_img_and_boxes(image, boxes, target_size):
    target_h = target_size[0]
    target_w = target_size[1]
    image_shape = image.size
    img_h = image_shape[0]
    img_w = image_shape[1]
    h_scale = target_h / img_h
    w_scale = target_w / img_w
    scale = np.minimum(h_scale, w_scale)
    new_h = int(img_h * scale)
    new_w = int(img_w * scale)
    pad_h_top = (target_h - new_h) // 2
    pad_h_bottom = target_h - new_h - pad_h_top
    pad_w_left = (target_w - new_w) // 2
    pad_w_right = target_w - new_w - pad_w_left
    image = image.resize((new_w, new_h), Image.ANTIALIAS)
    image = np.asarray(image)
    image = np.pad(image, [[pad_h_top, pad_h_bottom], [pad_w_left, pad_w_right], [0, 0]], mode='constant')
    boxes[:, 0] = (boxes[:, 0] * img_w * scale + pad_w_left) / target_w
    boxes[:, 2] = (boxes[:, 2] * img_w * scale + pad_w_left) / target_w
    boxes[:, 1] = (boxes[:, 1] * img_h * scale + pad_h_top) / target_h
    boxes[:, 3] = (boxes[:, 3] * img_h * scale + pad_h_top) / target_h
    image = Image.fromarray(image)
    return image, boxes



def dict_to_tf_example(image_path, data, label_map_dict):
    """
    Convert json derived dict to tf.Example proto
    Notice that this function normalizes the bounding box coordinates provided by the raw data
    :param image_path:
    :param data: dict holding json fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    :param label_map_dict: A map from string label names to integers ids.
    :return: example: The convertted tf.Example
    :raises: ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG' and image.format != 'PNG':
        raise ValueError('Image format not JPEG or PNG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width, height = image.size
    #if width != 1600 and height != 1200:
    #    print(width, height)
    image_format = os.path.splitext(image_path)[1]
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    difficult = []
    for vehicle in data['det_results']:
        anno = vehicle
        x_min = max(anno['x_min'], 0)
        y_min = max(anno['y_min'], 0)
        x_max = anno['x_max']
        y_max = anno['y_max']
        xmin.append(float(x_min) / width)
        ymin.append(float(y_min) / height)
        xmax.append(float(x_max) / width)
        ymax.append(float(y_max) / height)
        vehicle_category = vehicle['class_id']
        #print(vehicle_category)
        category_width = x_max - x_min
        vehicle_category = min(vehicle_category, 1)
        classes.append(vehicle_category + 1)
        if vehicle_category == 0:
            classes_text.append(bytes('head', encoding='utf-8'))
        else:
            classes_text.append(bytes('rear', encoding='utf-8'))
        if 'NotUse' in vehicle['types'] or category_width < 240:
            difficult.append(int(True))
        else:
            difficult.append(int(False))
    global pics, gts, simple, hard
    pics += 1
    gts += len(data['det_results'])
    simple += difficult.count(False)
    hard += difficult.count(True)
    #height = 240
    #width = 320
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    difficult = np.asarray(difficult, dtype=np.int32)
    classes = np.asarray(classes, dtype=np.int32)
    #target_size = [height, width]
    #image = image.resize((width, height), Image.ANTIALIAS)
    #image, boxes = transform_img_and_boxes(image, boxes, target_size)
    xmin = list(boxes[:, 0])
    ymin = list(boxes[:, 1])
    xmax = list(boxes[:, 2])
    ymax = list(boxes[:, 3])
    #image = image.resize((width, height), Image.ANTIALIAS)
    temp_io = io.BytesIO()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
               'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(image_path, encoding='utf-8')])),
                'boxes':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()])),
                'is_crowd': tf.train.Feature(bytes_list=tf.train.BytesList(value=[difficult.tostring()])),
                'class' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()])) 
            }))
    return example


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    logging.info('Reading list from %s' % FLAGS.list_path)
    with tf.gfile.GFile(FLAGS.list_path) as fid:
        lines = fid.readlines()
    logging.info('List load.')
    if FLAGS.shuffle:
        random.shuffle(lines)
    for idx, line in enumerate(tqdm(lines, dynamic_ncols=True)):
        example_path = line.strip()
        json_path = os.path.splitext(example_path)[0] + '.json'
        fp = open(json_path)
        data = json.load(fp)
        fp.close()
        if len(data['det_results']) == 0:
            continue
        tf_example = dict_to_tf_example(example_path, data, label_map_dict)
        if tf_example:
            writer.write(tf_example.SerializeToString())

    writer.close()
    print('Number of pics   = ', pics)
    print('Number of boxes  = ', gts)
    print('Number of simple boxes = ', simple)
    print('Number of hard   boxes = ', hard)
