#!/bin/bash

#train
# python run_piqa.py \
#   --model_name_or_path ../../../gpt2-medium \
#   --dataset_name ../../datasets/piqa \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 12 \
#   --save_steps 5000 \
#   --output_dir ../tmp/medium_xwwxsxv_piqa_default12 \
#   --overwrite_output_dir

#eval
python run_piqa.py \
  --model_name_or_path ../tmp/medium_ori_piqa_default12 \
  --dataset_name ../../datasets/piqa \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 12 \
  --save_steps 5000 \
  --output_dir ../tmp/medium_ori_piqa_100_eval