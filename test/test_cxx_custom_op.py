import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
# from tensorflow.python.platform import resource_loader
# from tensorflow.python.data.ops import dataset_ops
# from tensorflow.python.data.util import structure
# from tensorflow.python.framework import dtypes



# my_reader_dataset_module = tf.load_op_library("/Users/sunjiahe/CLionProjects/tf_ops/test.so")
my_reader_dataset_module = tf.load_op_library("/Users/sunjiahe/CLionProjects/tf_ops/cmake-build-release/libops.so")
# print(my_reader_dataset_module)
# a = tf.ones(shape=[3, 3], dtype=tf.int32)
# b = my_reader_dataset_module.zero_out(a)
# print(b)

class MyReaderDataset(tf.data.Dataset):
    def __init__(self):
        super(MyReaderDataset, self).__init__()

    def _inputs(self):
        return []

    def _as_variant_tensor(self):
        return my_reader_dataset_module.my_reader_dataset()

    @property
    def output_types(self):
        return tf.string

    @property
    def output_shapes(self):
        return tf.TensorShape([])

    @property
    def output_classes(self):
        return tf.Tensor

if __name__ == "__main__":
    # Create a MyReaderDataset and print its elements.
    with tf.Session() as sess:
        iterator = MyReaderDataset().make_one_shot_iterator()
        next_element = iterator.get_next()
        try:
            while True:
                print(sess.run(next_element))  # Prints "MyReader!" ten times.
        except tf.errors.OutOfRangeError:
            pass