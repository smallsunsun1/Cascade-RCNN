import yaml
import easydict

file = open("/Users/sunjiahe/PycharmProjects/master_work/config/rcnn_fpn.yaml")
config = yaml.safe_load(file)
a = easydict.EasyDict(config)
