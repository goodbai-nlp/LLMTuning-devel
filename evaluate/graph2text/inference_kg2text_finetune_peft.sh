#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BasePath=/baixuefeng
BaseModel=${BasePath}/data/pretrained-models/llama-7b

DATA=${BasePath}/data/AMRData/LDC2020-mtl-raw/test-NodeEdge.jsonl
DATA=${BasePath}/data/AMRData/LDC2020-mtl/test-Sent.jsonl
DATA=${BasePath}/data/AMRData/LDC2020-amr2text/test.jsonl
DATA=${BasePath}/data/AMRData/LDC2020-amr/test.jsonl
#DATA=${BasePath}/data/AMRData/LDC2020-ori-amr/test.jsonl
DATA=${BasePath}/data/KGData/webnlg/test.jsonl

PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
python -u inference_peft.py \
    --test_file ${DATA} \
    --base_model_name_or_path ${BaseModel} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_prefix "test-pred" \
    --input_key "src" \
    --instruction "Generate a descriptive text for the given knowledge graph." \
    --prompt_template "supervised" 2>&1 | tee $MODEL/eval.log