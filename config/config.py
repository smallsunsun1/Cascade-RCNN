import easydict
import yaml

config_file = "./config/rcnn_fpn.yaml"
yaml_config = yaml.safe_load(open(config_file))
_C = easydict.EasyDict(yaml_config)
