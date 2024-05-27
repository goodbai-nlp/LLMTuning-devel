#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BasePath=/home/export/base/ycsc_chenkh/chenkh_nvlink/online1/xfbai/
DATA=${BasePath}/data/AMRData/LDC2017-amrparsing-llama3/test.jsonl

PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
python -u ../inference_vllm.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 1024 \
    --out_prefix "pred" \
    --input_key "sent" \
    --instruction "Generate the abstract meaning graph for the given sentence." \
    --prompt_template "supervised-llama3" 2>&1 | tee $MODEL/eval.log
