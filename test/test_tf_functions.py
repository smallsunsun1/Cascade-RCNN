import tensorflow as tf
import sys
import numpy as np
sys.path.append("..")
sys.path.append(".")
from util import data_loader

if __name__ == '__main__':
    tf.enable_eager_execution()
    filename = "/home/admin-seu/TempData/sss/Master_work/data/train_coco.tfrecord"
    dataset = data_loader.input_fn(filename, False)
    for idx, ele in enumerate(dataset):
        #print(ele["class"])
        print(tf.where(tf.greater(ele['image'], 10)))
        if idx == 100:
            break
        # pass

