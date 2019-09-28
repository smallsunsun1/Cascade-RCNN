python train.py --train_filename="/mnt/WXRG0243/jhsun/Github/Master_work/data/train_face.tfrecord" \
  --eval_filename="/mnt/WXRG0243/jhsun/Github/Master_work/data/train_face.tfrecord" \
  --test_filename="/mnt/WXRG0243/jhsun/Github/Master_work/data/tutorial_eval.list" \
  --model=fpn \
  --gpus=8 \
  --model_dir="./fpn_cascade_face_model_v1"
