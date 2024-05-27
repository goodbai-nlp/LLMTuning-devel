#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL=$1
DATA=./test-data/test.json
DATA=./test-data/test-0513.json

python -u ../inference_vllm.py --test_file ${DATA} --model_name_or_path ${MODEL} --num_beams 1 --max_new_tokens 512 --out_prefix "Finetuned-7b-chat-pred" --instruction "" --input_key "Question" --prompt_template "llama2-chat" 2>&1 | tee $MODEL/eval.log
