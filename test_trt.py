import tensorflow as tf
import numpy as np
import time

# output_saved_model_dir = "./convert_export_model"
output_saved_model_dir = "./convert_INT8_export_model"
# output_saved_model_dir = "./convert_FP32_export_model"

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               output_saved_model_dir)
    cur_graph = sess.graph
    node_names = [tensor.name for tensor in sess.graph_def.node]
    output_tensors = []
    input_tensor = cur_graph.get_tensor_by_name("input:0")
    output_tensors.append(cur_graph.get_tensor_by_name("boxes:0"))
    output_tensors.append(cur_graph.get_tensor_by_name("scores_1:0"))
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
    start = time.time()
    for i in range(100):
        output = sess.run(output_tensors, feed_dict={input_tensor: np.zeros(shape=[800, 800, 3], dtype=np.float32)})
    end = time.time()
    print(end - start)
    print(output[1])
