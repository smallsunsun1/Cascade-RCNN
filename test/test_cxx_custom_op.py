import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops



# my_reader_dataset_module = tf.load_op_library("/Users/sunjiahe/CLionProjects/tf_ops/test2.so")
my_reader_dataset_module = tf.load_op_library("/Users/sunjiahe/CLionProjects/tf_ops/zero_out.so")
# my_reader_dataset_module = tf.load_op_library("/Users/sunjiahe/CLionProjects/tf_ops/cmake-build-release/libops.so")
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


@ops.RegisterGradient("ToZeros")
def _to_zeros_grad(op, grad):
    out_grad = array_ops.zeros_like(grad)
    return [out_grad]

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input

if __name__ == "__main__":
    tf.enable_eager_execution()
    # Create a MyReaderDataset and print its elements.
    # with tf.Session() as sess:
    #     iterator = MyReaderDataset().make_one_shot_iterator()
    #     next_element = iterator.get_next()
    #     try:
    #         while True:
    #             print(sess.run(next_element))  # Prints "MyReader!" ten times.
    #     except tf.errors.OutOfRangeError:
    #         pass
    a = tf.ones(shape=[3, 3], dtype=tf.int32)
    with tf.GradientTape() as g:
        g.watch(a)
        b = a * a
        g.watch(b)
    d = g.gradient(b, a)
    print(d)

