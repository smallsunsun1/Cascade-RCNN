import numpy as np
import tensorflow as tf

def py_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y2 + 1)
    ## order是按照score降序排序的
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

if __name__ == '__main__':
    tf.enable_eager_execution()
    p = [0.2, 0.3, 0.1, 0.4]
    dist = tf.distributions.Categorical(probs=p)
    index = dist.sample()
    value = tf.convert_to_tensor([[720, 720],
                                  [640, 640]])
    print(value[index, :])