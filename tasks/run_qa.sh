#!/bin/bash

# python run_qa.py \
#   --model_name_or_path ../../../Meta-Llama-3-8B-Instruct \
#   --dataset_name ../../datasets/squad \
#   --do_eval \
#   --per_device_train_batch_size 1 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 256 \
#   --doc_stride 128 \
#   --save_steps 5000 \
#   --output_dir ../tmp/medium-ori-squad/medium_ori_squad_99_eval \
#   --bf16 

# torchrun --nproc_per_node=4 run_qa.py \
#   --model_name_or_path ../../../gpt2-medium \
#   --dataset_name ../../datasets/squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 16 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --save_steps 5000 \
#   --output_dir ../tmp/medium-ori-squad/medium_ori_squad_99_eval


#train
# python run_qa.py \
#   --model_name_or_path ../../../gpt2-medium \
#   --dataset_name ../../datasets/squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --save_steps 5000 \
#   --output_dir ../tmp/medium-ori-squad/medium_ori_squad_99_eval 


#eval
python run_qa.py \
  --model_name_or_path ../tmp/medium-ori-squad/medium_ori_squad \
  --dataset_name ../../datasets/squad \
  --do_eval \
  --per_device_eval_batch_size 12 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 5000 \
  --output_dir ../tmp/medium-ori-squad/medium_ori_squad_100_eval___  