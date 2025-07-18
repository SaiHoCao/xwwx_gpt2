#!/bin/bash

# python run_qa.py \
#   --model_name_or_path ../../gpt2-medium \
#   --dataset_name ../datasets/squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --save_steps 5000 \
#   --output_dir ./tmp/medium_squad_ori_10 \
#   --version_2_with_negative \

torchrun --nproc_per_node=4 run_qa.py \
  --model_name_or_path ./tmp/medium_squad_ori \
  --dataset_name ../datasets/squad \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 5000 \
  --output_dir ./tmp/0_medium_squad_ori_100_eval

# torchrun --nproc_per_node=4 run_qa.py \
#   --model_name_or_path ../../gpt2-medium \
#   --dataset_name ../datasets/squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 16 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --save_steps 5000 \
#   --output_dir ./tmp/medium_squad_ori