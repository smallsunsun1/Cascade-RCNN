import tensorflow as tf
import sys
import numpy as np
sys.path.append("..")

from util import data_loader

if __name__ == '__main__':
    tf.enable_eager_execution()
    filename = "/home/admin-seu/sss/master_work/data/train_voco.tfrecord"
    dataset = data_loader.input_fn(filename, False)
    for idx, ele in enumerate(dataset):
        print(np.sum(ele["anchor_labels"] + 1))
        # pass

