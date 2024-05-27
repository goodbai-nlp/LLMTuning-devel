#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export PATH="/opt/conda/bin:$PATH"
#source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

BasePath=/UNICOMFS/hitsz_khchen_1
PORT_ID=$(expr $RANDOM + 1000)
datacate=webnlg17
datacate=webnlg20
datacate=EventNarrative
datacate=webnlg17-llama3

DATA=${BasePath}/data/Data2text/${datacate}/test.jsonl

MODEL=$1

python -u inference_hf.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_prefix "test-pred" \
    --input_key "input" \
    --instruction "Generate a coherent piece of text that contains all of the information in the triples." \
    --prompt_template "chat-v1" 2>&1 | tee $MODEL/eval.log
