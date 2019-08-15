import tensorflow as tf
import sys
sys.path.append("..")

from util import data_loader

if __name__ == '__main__':
    tf.enable_eager_execution()
    filename = "/home/admin-seu/sss/master_work/data/train.record"
    dataset = data_loader.input_fn(filename, False)
    for idx, ele in enumerate(dataset):
        mat = ele['boxes'].numpy()
        if mat.shape[0] == 0:
            print("here")
        if idx % 500 == 0:
            print(idx)

