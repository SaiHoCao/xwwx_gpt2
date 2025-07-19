#!/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python run_clm.py \
#     --model_name_or_path ../../Meta-Llama-3-8B-Instruct \
#     --dataset_name ../datasets/wikitext-2-raw-v1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_checkpointing \
#     --block_size 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 3 \
#     --save_steps 2000 \
#     --output_dir ./tmp/medium-ori-wiki2/medium_ori_wiki2_99_eval \
#     --bf16

# torchrun --nproc_per_node=4 run_clm.py \
#     --model_name_or_path ../../gpt2-medium \
#     --dataset_name ../datasets/wikitext-2-raw-v1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 3 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 3 \
#     --save_steps 2000 \
#     --output_dir ./tmp/medium-ori-wiki2/medium_ori_wiki2_99_eval


#train
# python run_clm.py \
#     --model_name_or_path ../../gpt2-medium \
#     --dataset_name ../datasets/wikitext-2-raw-v1 \
#     --per_device_train_batch_size 6 \
#     --per_device_eval_batch_size 6 \
#     --do_train \
#     --do_eval \
#     --save_steps 2000 \
#     --output_dir ./tmp/medium-ori-wiki2/medium_ori_wiki2 


#eval
python run_clm.py \
    --model_name_or_path ./tmp/medium-ori-wiki2/medium_ori_wiki2 \
    --dataset_name ../datasets/wikitext-2-raw-v1 \
    --per_device_eval_batch_size 12 \
    --do_eval \
    --save_steps 2000 \
    --output_dir ./tmp/medium-ori-wiki2/medium_ori_wiki2_100_eval