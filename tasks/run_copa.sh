#!/bin/bash

#train
# python run_copa.py \
#   --model_name_or_path ../../../gpt2-medium \
#   --dataset_name ../../datasets/balanced-copa \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 24 \
#   --save_steps 5000 \
#   --output_dir ../tmp/medium-xwwxsxv-copa/medium_xwwxsxv_copa_99_eval

#eval
python run_copa.py \
  --model_name_or_path ../tmp/medium-xwwxsxv-copa/medium_xwwxsxv_copa \
  --dataset_name ../../datasets/balanced-copa \
  --do_eval \
  --per_device_eval_batch_size 12 \
  --save_steps 5000 \
  --output_dir ../tmp/medium-xwwxsxv-copa/medium_xwwxsxv_copa_100_eval