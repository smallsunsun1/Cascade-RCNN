import tensorflow as tf
import cv2
import base64
from tensorflow.python.compiler.tensorrt import trt_convert as trt

saved_model_dir = "./export_model_0126/1581080318"
output_saved_model_dir = "./convert_INT8_export_model"
fetch_names = ["strided_slice_256:0", "cond/Merge:0", "strided_slice_1:0", "ExpandDims:0",
               "Const_39:0", "strided_slice_258:0", "Shape:0", "Cast:0",
               "strided_slice_260:0", "combined_non_max_suppression/CombinedNonMaxSuppression:3",
               "cond/Merge_1:0", "strided_slice_2:0"]

class feed_dict_input_fn():
    def __init__(self, filename):
        self.filename = filename
        self.content = []
        with open(self.filename) as f:
            for line in f:
                self.content.append(line.strip())
        self.index = 0
    def __call__(self, *args, **kwargs):
        data = open(self.content[self.index], 'rb').read()
        encode = base64.urlsafe_b64encode(data)
        encode = str(encode, encoding='utf-8')
        image = {"input:0": encode}
        # value = {"inputs": image}
        self.index += 1
        return image


converter = trt.TrtGraphConverter(input_saved_model_dir=saved_model_dir,
                                  precision_mode=trt.TrtPrecisionMode.INT8,
                                  use_calibration=True, is_dynamic_op=True, maximum_cached_engines=3)
feet_dict_input = feed_dict_input_fn("/home/admin-seu/TempData/sss/Master_work/data/test.list")
converter.convert()
converter.calibrate(fetch_names=fetch_names, num_runs=100, feed_dict_fn=feet_dict_input)
converter.save(output_saved_model_dir)
