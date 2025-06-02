#!/bin/bash

python run_clm.py \
    --model_name_or_path ../../gpt2 \
    --dataset_name ../datasets/wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 10000 \
    --output_dir ./tmp/test-clm-xwwx-sxv-scores