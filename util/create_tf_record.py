import tensorflow as tf
import numpy as np
import os
import re
import json

def generate(input_filename, output_filename):
    path = output_filename
    writer = tf.io.TFRecordWriter(path)
    for ele in open(input_filename):
        ele = ele.strip()
        ele = re.sub(",", " ", ele)
        ele = ele.split(" ")
        filename = ele[0]
        boxes = np.asarray(ele[1:]).astype(np.float32).reshape([-1, 5])
        if np.shape(boxes)[0] == 0:
            print("no element !")
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


def generate_from_voco(input_file, output_file):
    path = output_file
    writer = tf.io.TFRecordWriter(path)
    filenames = open(input_file).readlines()
    for filename in filenames:
        filename = filename.strip()
        feature = {}
        feature["filename"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, encoding="utf-8")]))
        info = re.split("/JPEGImages/", filename)
        label_file = os.path.join(info[0], "labels", info[1].split('.')[0] + ".txt")
        label_info = open(label_file).readlines()
        boxes = []
        classes = []
        for label_in in label_info:
            label_in = label_in.strip()
            element = label_in.split(" ")
            box = element[1:]
            Class = element[0]
            box = np.asarray(box, np.float32)
            Class = np.asarray(Class, np.int32)
            new_box = []
            new_box.append(box[0] - box[2]/2)
            new_box.append(box[1] - box[3]/2)
            new_box.append(box[0] + box[2]/2)
            new_box.append(box[1] + box[3]/2)
            #print(new_box)
            boxes.append(new_box)
            classes.append(Class)
        boxes = np.asarray(boxes, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int32)
        is_crowd = np.zeros(shape=[np.shape(boxes)[0], ], dtype=np.int32)
        feature["boxes"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
        feature["is_crowd"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_crowd.tostring()]))
        feature["class"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()]))
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()


def generate_from_json(in_file, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    with open(in_file) as f:
        for ele in f:
            ele = ele.strip()
            filename = ele
            json_filename = ele.strip('.jpg') + '.json'
            json_data = json.load(open(json_filename))
            boxes = []
            classes = []
            is_crowd = []
            for anno in json_data['annotation']:
                box = anno['bbox']
                box[2] = box[2] + box[0]
                box[3] = box[3] + box[1]
                box[0] /= json_data['image']['width']
                box[2] /= json_data['image']['width']
                box[1] /= json_data['image']['height']
                box[3] /= json_data['image']['height']
                boxes.append(box)
                classes.append(0)
                is_crowd.append(anno['iscrowd'])
            boxes = np.asarray(boxes, dtype=np.float32)
            classes = np.asarray(classes, dtype=np.int32)
            is_crowd = np.asarray(is_crowd, dtype=np.int32)
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
    # input_filenames = "/home/admin-seu/hugh/yolov3-tf2/data_native/eval.txt"
    # input_filenames = "/home/admin-seu/sss/master_work/data/train.txt"
    #input_filenames = "/mnt/WXRG0243/jhsun/Data/2007_test.txt"
    input_filenames = "/mnt/WXRG0243/jhsun/Data/train.txt"
    # output_filenames = "/home/admin-seu/sss/master_work/data/eval.record"
    output_filenames = '/mnt/WXRG0243/jhsun/Github/Master_work/data/train_voco.tfrecord'
    #output_filenames = "/mnt/WXRG0243/jhsun/Github/Master_work/data/eval_voco.tfrecord"
    generate_from_voco(input_filenames, output_filenames)

    #input_filenames = "/mnt/ficuspi/ybkang/data/camp/tutorial_train.list"
    #input_filenames = "/mnt/WXRG0243/jhsun/Github/Master_work/data/tutorial_eval.list"
    #output_filenames = "/mnt/WXRG0243/jhsun/Github/Master_work/data/train_face.tfrecord"
    #generate_from_json(input_filenames, output_filenames)
    dataset = input_fn(output_filenames)
    for ele in dataset:
       print(ele['boxes'])
