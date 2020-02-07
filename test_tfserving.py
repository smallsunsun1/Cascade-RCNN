import requests
import json
import io
import cv2
import tensorflow as tf
import numpy as np
import base64

def map_boxes_back(boxes, features):
    h_pre = features['h_pre']
    w_pre = features['w_pre']
    h_now = features['h_now']
    w_now = features['w_now']
    scale = features['scale']
    if scale > 1:
        true_h = w_now / scale
        pad_h_top = (h_now - true_h) // 2
        pad_w_left = 0
        true_w = w_now
    else:
        true_w = h_now * scale
        pad_w_left = (w_now - true_w) // 2
        pad_h_top = 0
        true_h = h_now
    boxes[:, 0] = boxes[:, 0] - pad_w_left
    boxes[:, 1] = boxes[:, 1] - pad_h_top
    boxes[:, 2] = boxes[:, 2] - pad_w_left
    boxes[:, 3] = boxes[:, 3] - pad_h_top
    boxes[:, 0] = boxes[:, 0] / true_w * w_pre
    boxes[:, 1] = boxes[:, 1] / true_h * h_pre
    boxes[:, 2] = boxes[:, 2] / true_w * w_pre
    boxes[:, 3] = boxes[:, 3] / true_h * h_pre
    return boxes


url = "http://121.248.54.238:8501/v1/models/cascade_rcnn:predict"
data = open("/Users/sunjiahe/PycharmProjects/CycleGan/MergePic/timg.jpeg", 'rb').read()
encode = base64.urlsafe_b64encode(data)
encode = str(encode, encoding='utf-8')
image_in = {"input": encode}
value = {"inputs": image_in}
response = requests.post(url, json=value)
res = json.loads(response.content)
res = res["outputs"]

total_res = []
score_thresh = 0.5
res["boxes"] = np.asarray(res["boxes"])
# res["boxes"] = map_boxes_back(res["boxes"], res)
image = np.squeeze(np.asarray(res["image"]), axis=0).astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for num_idx, box in enumerate(res["boxes"]):
    if res["scores"][num_idx] < score_thresh:
        continue
    info_dict = {}
    info_dict["image_id"] = int(res["image_id"])
    info_dict["category_id"] = int(res["labels"][num_idx])
    info_dict["bbox"] = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
    info_dict["score"] = float(res["scores"][num_idx])
    total_res.append(info_dict)
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.putText(image, '{}: {:.2}'.format(res["labels"][num_idx], round(res["scores"][num_idx], 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite("./res.jpg", image)
