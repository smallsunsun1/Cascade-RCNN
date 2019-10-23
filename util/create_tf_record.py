import tensorflow as tf
import numpy as np
import os
import re
import json
import tqdm

from PIL import Image
from dataset import DatasetSplit

def generate(input_filename, output_filename):
    path = output_filename
    writer = tf.io.TFRecordWriter(path)
    for ele in open(input_filename):
        ele = ele.strip()
        ele = re.sub(",", " ", ele)
        ele = ele.split(" ")
        filename = ele[0]
        boxes = np.asarray(ele[1:]).astype(np.float32).reshape([-1, 5])
        if np.shape(boxes)[0] == 0:
            print("no element !")
        classes = boxes[:, 4].astype(np.int32)
        boxes = boxes[:, :4].astype(np.float32)
        is_crowd = np.zeros(shape=[np.shape(boxes)[0], ], dtype=np.int32)
        feature = {}
        feature["filename"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, encoding="utf-8")]))
        feature["boxes"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
        feature["is_crowd"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_crowd.tostring()]))
        feature["class"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()]))
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()


def generate_from_voco(input_file, output_file):
    path = output_file
    writer = tf.io.TFRecordWriter(path)
    filenames = open(input_file).readlines()
    for filename in filenames:
        filename = filename.strip()
        feature = {}
        feature["filename"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, encoding="utf-8")]))
        info = re.split("/JPEGImages/", filename)
        label_file = os.path.join(info[0], "labels", info[1].split('.')[0] + ".txt")
        label_info = open(label_file).readlines()
        boxes = []
        classes = []
        for label_in in label_info:
            label_in = label_in.strip()
            element = label_in.split(" ")
            box = element[1:]
            Class = element[0]
            box = np.asarray(box, np.float32)
            Class = np.asarray(Class, np.int32)
            new_box = []
            new_box.append(box[0] - box[2]/2)
            new_box.append(box[1] - box[3]/2)
            new_box.append(box[0] + box[2]/2)
            new_box.append(box[1] + box[3]/2)
            #print(new_box)
            boxes.append(new_box)
            classes.append(Class)
        boxes = np.asarray(boxes, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int32)
        is_crowd = np.zeros(shape=[np.shape(boxes)[0], ], dtype=np.int32)
        feature["boxes"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
        feature["is_crowd"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_crowd.tostring()]))
        feature["class"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()]))
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()


def generate_from_json(in_file, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    with open(in_file) as f:
        for ele in f:
            ele = ele.strip()
            filename = ele
            json_filename = ele.strip('.jpg') + '.json'
            json_data = json.load(open(json_filename))
            boxes = []
            classes = []
            is_crowd = []
            for anno in json_data['annotation']:
                box = anno['bbox']
                box[2] = box[2] + box[0]
                box[3] = box[3] + box[1]
                box[0] /= json_data['image']['width']
                box[2] /= json_data['image']['width']
                box[1] /= json_data['image']['height']
                box[3] /= json_data['image']['height']
                boxes.append(box)
                classes.append(0)
                is_crowd.append(anno['iscrowd'])
            boxes = np.asarray(boxes, dtype=np.float32)
            classes = np.asarray(classes, dtype=np.int32)
            is_crowd = np.asarray(is_crowd, dtype=np.int32)
            feature = {}
            feature["filename"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, encoding="utf-8")]))
            feature["boxes"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
            feature["is_crowd"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_crowd.tostring()]))
            feature["class"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()]))
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_string = example.SerializeToString()
            writer.write(example_string)
    writer.close() 


class COCODetection(DatasetSplit):
    # handle a few special splits whose names do not match the directory names
    _INSTANCE_TO_BASEDIR = {
        'valminusminival2014': 'val2014',
        'minival2014': 'val2014',
        'val2017_100': 'val2017',
    }

    """
    Mapping from the incontinuous COCO category id to an id in [1, #category]
    For your own coco-format, dataset, change this to an **empty dict**.
    """
    COCO_id_to_category_id = {13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}  # noqa

    def __init__(self, basedir, split):
        """
        Args:
            basedir (str): root of the dataset which contains the subdirectories for each split and annotations
            split (str): the name of the split, e.g. "train2017".
                The split has to match an annotation file in "annotations/" and a directory of images.

        Examples:
            For a directory of this structure:

            DIR/
              annotations/
                instances_XX.json
                instances_YY.json
              XX/
              YY/

            use `COCODetection(DIR, 'XX')` and `COCODetection(DIR, 'YY')`
        """
        basedir = os.path.expanduser(basedir)
        self._imgdir = os.path.realpath(os.path.join(
            basedir, self._INSTANCE_TO_BASEDIR.get(split, split)))
        assert os.path.isdir(self._imgdir), "{} is not a directory!".format(self._imgdir)
        annotation_file = os.path.join(
            basedir, 'annotations/instances_{}.json'.format(split))
        assert os.path.isfile(annotation_file), annotation_file

        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file)
        self.annotation_file = annotation_file
        print("Instances loaded from {}.".format(annotation_file))

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    def print_coco_metrics(self, results):
        """
        Args:
            results(list[dict]): results in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        has_mask = "segmentation" in results[0]  # results will be modified by loadRes

        cocoDt = self.coco.loadRes(results)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        if len(results) > 0 and has_mask:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        img_ids = self.coco.getImgIds()
        img_ids.sort()
        # list of dict, each has keys: height,width,id,file_name
        imgs = self.coco.loadImgs(img_ids)

        for idx, img in enumerate(tqdm.tqdm(imgs)):
            img['image_id'] = img.pop('id')
            img['file_name'] = os.path.join(self._imgdir, img['file_name'])
            if idx == 0:
                # make sure the directories are correctly set
                assert os.path.isfile(img["file_name"]), img["file_name"]
            if add_gt:
                self._add_detection_gt(img, add_mask)
        return imgs

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['image_id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['image_id']]  # equivalent but faster than the above two lines
        if 'minival' not in self.annotation_file:
            # TODO better to check across the entire json, rather than per-image
            ann_ids = [ann["id"] for ann in objs]
            assert len(set(ann_ids)) == len(ann_ids), \
                "Annotation ids in '{}' are not unique!".format(self.annotation_file)

        # clean-up boxes
        width = img.pop('width')
        height = img.pop('height')

        all_boxes = []
        all_segm = []
        all_cls = []
        all_iscrowd = []
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = list(map(float, obj['bbox']))
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel
            x2, y2 = x1 + w, y1 + h

            # np.clip would be quite slow here
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)
            w, h = x2 - x1, y2 - y1
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and w > 0 and h > 0:
                all_boxes.append([x1, y1, x2, y2])
                all_cls.append(self.COCO_id_to_category_id.get(obj['category_id'], obj['category_id']))
                iscrowd = obj.get("iscrowd", 0)
                all_iscrowd.append(iscrowd)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert iscrowd == 1
                        all_segm.append(None)
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) == 0:
                            logger.error("Object {} in image {} has no valid polygons!".format(objid, img['file_name']))
                        elif len(valid_segs) < len(segs):
                            logger.warn("Object {} in image {} has invalid polygons!".format(objid, img['file_name']))
                        all_segm.append(valid_segs)

         # all geometrically-valid boxes are returned
        if len(all_boxes):
            img['boxes'] = np.asarray(all_boxes, dtype='float32')  # (n, 4)
        else:
            img['boxes'] = np.zeros((0, 4), dtype='float32')
        cls = np.asarray(all_cls, dtype='int32')  # (n,)
        if len(cls):
            assert cls.min() > 0, "Category id in COCO format must > 0!"
        img['class'] = cls          # n, always >0
        img['is_crowd'] = np.asarray(all_iscrowd, dtype='int8')  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = all_segm

    def training_roidbs(self):
        return self.load(add_gt=True, add_mask=cfg.MODE_MASK)

    def inference_roidbs(self):
        return self.load(add_gt=False)

    def eval_inference_results(self, results, output=None):
        continuous_id_to_COCO_id = {v: k for k, v in self.COCO_id_to_category_id.items()}
        for res in results:
            # convert to COCO's incontinuous category id
            if res['category_id'] in continuous_id_to_COCO_id:
                res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
            # COCO expects results in xywh format
            box = res['bbox']
            box[2] -= box[0]
            box[3] -= box[1]
            res['bbox'] = [round(float(x), 3) for x in box]

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            return self.print_coco_metrics(results)
        else:
            return {}


def generate_from_coco_json(basedir,  year='train2017', output_file='./data/coco_train.tfrecord'):
    coco_detection =COCODetection(basedir, year)
    roidbs = coco_detection.load(add_gt=True, add_mask=False)
    writer = tf.io.TFRecordWriter(output_file)
    for idx, element in enumerate(tqdm.tqdm(roidbs)):
        boxes = element['boxes']
        classes = element['class'].astype(np.int32)
        is_crowd = element['is_crowd'].astype(np.int32)
        filename = element['file_name']
        img = Image.open(filename)
        width, height = img.size
        boxes[:, 0::2] = boxes[:, 0::2] / width
        boxes[:, 1::2] = boxes[:, 1::2] / height
        feature = {}
        feature['filename'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, encoding='utf-8')]))
        feature['boxes'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes.tostring()]))
        feature['is_crowd'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_crowd.tostring()]))
        feature['class'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes.tostring()]))
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()


def parse_raw(features):
    features["boxes"] = tf.reshape(tf.io.decode_raw(features["boxes"], tf.float32), shape=[-1, 4])
    features["class"] = tf.reshape(tf.decode_raw(features['class'], tf.int32), shape=[-1])
    features["is_crowd"] = tf.reshape(tf.decode_raw(features['is_crowd'], tf.int32), shape=[-1])
    # image = tf.io.read_file(features["filename"])
    # features["image"] = image
    return features


def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(x, features={"filename": tf.io.FixedLenFeature([], tf.string),
                                                          "boxes": tf.io.FixedLenFeature([], tf.string),
                                                          "class": tf.io.FixedLenFeature([],
                                                                                         tf.string),
                                                          "is_crowd": tf.io.FixedLenFeature([],
                                                                                            tf.string)
                                                          }))
    dataset = dataset.map(lambda x: parse_raw(x))
    return dataset





if __name__ == "__main__":
    tf.enable_eager_execution()
    # input_filenames = "/home/admin-seu/hugh/yolov3-tf2/data_native/eval.txt"
    # input_filenames = "/home/admin-seu/sss/master_work/data/train.txt"
    #input_filenames = "/mnt/WXRG0243/jhsun/Data/2007_test.txt"
    input_filenames = "/mnt/WXRG0243/jhsun/Data/train.txt"
    # output_filenames = "/home/admin-seu/sss/master_work/data/eval.record"
    output_filenames = '/mnt/WXRG0243/jhsun/Github/Master_work/data/train_voco.tfrecord'
    #output_filenames = "/mnt/WXRG0243/jhsun/Github/Master_work/data/eval_voco.tfrecord"
    #generate_from_voco(input_filenames, output_filenames)

    #input_filenames = "/mnt/ficuspi/ybkang/data/camp/tutorial_train.list"
    #input_filenames = "/mnt/WXRG0243/jhsun/Github/Master_work/data/tutorial_eval.list"
    #output_filenames = "/mnt/WXRG0243/jhsun/Github/Master_work/data/train_face.tfrecord"
    #generate_from_json(input_filenames, output_filenames)

    ## This part is used for coco data loading test
    basedir = '/mnt/WXRG0243/jhsun/Data'
    year = 'val2017'
    output_filenames = '/mnt/WXRG0243/jhsun/Github/Master_work/data/eval_coco.tfrecord'
    generate_from_coco_json(basedir, year, output_filenames)
    #roidbs = c.load(add_gt=True, add_mask=False)
    #for ele in roidbs[:10]:
    #    print(ele)
    #    print()
    #dataset = input_fn(output_filenames)
    #for ele in dataset:
    #   print(ele.keys())    
    #   print(ele['boxes'])
