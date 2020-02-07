import tensorflow as tf
import numpy as np
import base64
import time

# output_saved_model_dir = "./convert_export_model"
# output_saved_model_dir = "./export_model_0126/1581080318"
output_saved_model_dir = "./convert_INT8_export_model"
# output_saved_model_dir = "./convert_FP32_export_model"

data = open("/home/admin-seu/TempData/test2017/000000258074.jpg", 'rb').read()
encode = base64.urlsafe_b64encode(data)
encode = str(encode, encoding='utf-8')

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               output_saved_model_dir)
    cur_graph = sess.graph
    node_names = [tensor.name for tensor in sess.graph_def.node]
    output_tensors = []
    input_tensor = cur_graph.get_tensor_by_name("input:0")
    output_tensors.append(cur_graph.get_tensor_by_name("strided_slice_256:0"))
    output_tensors.append(cur_graph.get_tensor_by_name("strided_slice_260:0"))
    # for node_name in node_names:
    #     if "input" in node_name:
    #         print(node_name)
    #         input_tensor = cur_graph.get_tensor_by_name(node_name)
    #     if "scores_1" in node_name:
    #         print(node_name)
    #         output_tensors.append(cur_graph.get_tensor_by_name(node_name))
    #     if "labels" in node_name:
    #         print(node_name)
    #         output_tensors.append(cur_graph.get_tensor_by_name(node_name))
    #     if "all_ids" in node_name:
    #         print(node_name)
    #         output_tensors.append(cur_graph.get_tensor_by_name(node_name))
    #     if "boxes_1" in node_name:
    #         print(node_name)
    #         output_tensors.append(cur_graph.get_tensor_by_name(node_name))
    output = sess.run(output_tensors, feed_dict={input_tensor: encode})
    print(np.shape(output[0]))
    for i in range(10):
        output = sess.run(output_tensors, feed_dict={input_tensor: encode})
    start = time.time()
    print(start)
    for i in range(100):
        output = sess.run(output_tensors, feed_dict={input_tensor: encode})
    end = time.time()
    print(end)
    print(end - start)

