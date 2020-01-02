import tensorflow as tf
import numpy as np

from tensorflow.contrib.factorization import KMeansClustering

def input_fn(array):
    return tf.train.limit_epochs(tf.convert_to_tensor(array, tf.float32), num_epochs=1)

