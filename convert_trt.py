import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

saved_model_dir = "./export_model/1578997166"
output_saved_model_dir = "./convert_INT8_export_model"
# output_saved_model_dir = "./convert_FP32_export_model"

converter = trt.TrtGraphConverter(input_saved_model_dir=saved_model_dir,
                                  precision_mode=trt.TrtPrecisionMode.INT8,
                                  maximum_cached_engines=100, use_calibration=False)
converter.convert()
converter.save(output_saved_model_dir)
