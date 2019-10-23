python train.py --train_filename="/mnt/WXRG0243/jhsun/Github/Master_work/data/train_voco.tfrecord" \
  --eval_filename="/mnt/WXRG0243/jhsun/Github/Master_work/data/eval_voco.tfrecord" \
  --test_filename="/mnt/WXRG0243/jhsun/Data/2007_test.txt" \
  --model=fpn \
  --gpus=3 \
  --model_dir="./fpn_cascade_voc_model_v2"
