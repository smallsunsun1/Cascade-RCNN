python train.py --train_filename="/home/admin-seu/TempData/sss/Master_work/data/train_coco.tfrecord" \
  --eval_filename="/home/admin-seu/TempData/sss/Master_work/data/eval_coco.tfrecord" \
  --test_filename="/home/admin-seu/TempData/sss/Master_work/data/test.list" \
  --model=fpn \
  --gpus=3 \
  --model_dir="./fpn_cascade_coco_model_sharedhead_v2"
