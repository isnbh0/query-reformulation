#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

poetry run accelerate launch --num_processes=2 train.py \
    --model_name t5-base \
    --train_dataset ./rsc/preprocessed/topiOCQA/train.json \
    --decode_type reformulation \
    --config_file configs/t5-base.yaml
