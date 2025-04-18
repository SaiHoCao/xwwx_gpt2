#!/bin/bash

python run_gpt2_clm_lm.py \
    --model_name_or_path ../../gpt2 \
    --dataset_name ../wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 96 \
    --save_steps 2000 \
    --output_dir ./tmp/test-clm