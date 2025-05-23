#!/bin/bash

python run_qa.py \
  --model_name_or_path ../../gpt2 \
  --dataset_name ../squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./tmp/debug_squad_xwwx