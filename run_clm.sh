#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_clm.py \
    --model_name_or_path ../../Meta-Llama-3-8B-Instruct \
    --dataset_name ../datasets/wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing \
    --block_size 512 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 2000 \
    --output_dir ./tmp/llama3_wiki2_RoPE \
    --bf16

# torchrun --nproc_per_node=4 run_clm.py \
#     --model_name_or_path ../../gpt2-medium \
#     --dataset_name ../datasets/wikitext-2-raw-v1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 3 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 3 \
#     --save_steps 2000 \
#     --output_dir ./tmp/test-clm-xwwx-medium-10