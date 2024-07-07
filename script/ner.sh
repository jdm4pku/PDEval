#!/bin/bash

# 定义命令行参数
TASK="entity"
TRAIN_DIR="/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/fold_0/train_data.json"
TEST_DIR="/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/fold_0/test_data.json"
SHOT_DIR="/home/jindongming/project/modeling/PDEval/input/fold_0/shot.json"
SHOT_NUM="1" #2,3,4,5
PROMPT_DIR="/home/jindongming/project/modeling/PDEval/prompt/ner.txt"
MODEL="llama3-8b" #qwen2-7b,glm4-9b,gemma-7b,llama3-8b,
MODA="greedy"
OUTPUT_DIR="/home/jindongming/project/modeling/PDEval/output"

# 运行Python脚本
CUDA_VISIBLE_DEVICES=0 python src/llm/llm_inference.py \
    --task $TASK \
    --train_dir $TRAIN_DIR \
    --test_dir $TEST_DIR \
    --shot_dir $SHOT_DIR \
    --shot_num $SHOT_NUM \
    --prompt_dir $PROMPT_DIR \
    --model $MODEL \
    --moda $MODA \
    --output_dir $OUTPUT_DIR