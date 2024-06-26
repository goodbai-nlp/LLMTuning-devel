#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
# export PATH="/opt/conda/bin:$PATH"
# source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

MODEL=$1
DATA=./test-data/test.json
DATA=./test-data/test-0513.json

python -u ../inference_vllm.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_path "./output" \
    --out_prefix "gemma-7b-it" \
    --instruction "" \
    --input_key "Question" \
    --prompt_template "gemma-it" 2>&1 | tee $MODEL/eval.log
