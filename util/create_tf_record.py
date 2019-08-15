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
        classes = boxes[:, 4].astype(np.int32)
        boxes = boxes[:, :4].astype(np.float32)
        is_crowd = np.zeros(shape=[np.shape(boxes)[0], ], dtype=np.int32)
        feature = {}
        feature["filename"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, encoding="utf-8")]))
        feature["boxes"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
        feature["is_crowd"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_crowd.tostring()]))
        feature["class"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()]))
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()


def parse_raw(features):
    features["boxes"] = tf.reshape(tf.io.decode_raw(features["boxes"], tf.float32), shape=[-1, 4])
    features["class"] = tf.reshape(tf.decode_raw(features['class'], tf.int32), shape=[-1])
    features["is_crowd"] = tf.reshape(tf.decode_raw(features['is_crowd'], tf.int32), shape=[-1])
    # image = tf.io.read_file(features["filename"])
    # features["image"] = image
    return features


def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(x, features={"filename": tf.io.FixedLenFeature([], tf.string),
                                                          "boxes": tf.io.FixedLenFeature([], tf.string),
                                                          "class": tf.io.FixedLenFeature([],
                                                                                         tf.string),
                                                          "is_crowd": tf.io.FixedLenFeature([],
                                                                                            tf.string)
                                                          }))
    dataset = dataset.map(lambda x: parse_raw(x))
    return dataset


if __name__ == "__main__":
    tf.enable_eager_execution()
    generate()
    dataset = input_fn("/Users/sunjiahe/PycharmProjects/master_work/data/train.record")
    for ele in dataset:
        print(ele)
