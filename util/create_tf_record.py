import tensorflow as tf
import numpy as np
import cv2
import re

def generate():
    path = "/Users/sunjiahe/PycharmProjects/master_work/data/train.record"
    writer = tf.io.TFRecordWriter(path)
    input_filename = "/Users/sunjiahe/Downloads/Windows/data_with_veri/train.txt"
    for ele in open(input_filename):
        ele = ele.strip()
        ele = re.sub(",", " ", ele)
        ele = ele.split(" ")
        filename = ele[0]
        boxes = np.asarray(ele[1:]).astype(np.float32).reshape([-1, 5])
        feature = {}
        feature["filename"] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[bytes(filename, encoding="utf-8")]))
        feature["boxes"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()

def parse_raw(features):
    features["boxes"] = tf.reshape(tf.io.decode_raw(features["boxes"], tf.float32), shape=[-1, 5])
    # image = tf.io.read_file(features["filename"])
    # features["image"] = image
    return features

def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    dataset = dataset.map(lambda x:tf.io.parse_single_example(x, features={"filename": tf.io.FixedLenFeature([], tf.string),
                                                                           "boxes": tf.io.FixedLenFeature([], tf.string)}))
    dataset = dataset.map(lambda x: parse_raw(x))
    return dataset

dataset = input_fn("/Users/sunjiahe/PycharmProjects/master_work/data/train.record")
for ele in dataset:
    print(ele)










