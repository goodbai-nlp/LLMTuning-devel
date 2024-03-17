#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export PATH="/opt/conda/bin:$PATH"
#source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

BasePath=/baixuefeng
DATA=${BasePath}/data/AMRData/LDC2020/test.jsonl
PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
python -u inference_vllm.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_prefix "pred" \
    --input_key "amr" \
    --instruction "Generate a descriptive text for the given graph." \
    --prompt_template "five_shot_amr" 2>&1 | tee $MODEL/eval.log
python eval_bleu.py ${DATA} output/pred_test_vllm