#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=1
#export PATH="/opt/conda/bin:$PATH"
#source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

#BasePath=/baixuefeng
PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
DATA=$2
python -u inference_hf.py --test_file ${DATA} --model_name_or_path ${MODEL} --num_beams 1 --max_new_tokens 512 --out_prefix "test-pred" --input_key "amr" --instruction "Generate a descriptive text for the given abstract meaning representation graph." --prompt_template "pubmedqa" 2>&1 | tee $MODEL/eval.log
