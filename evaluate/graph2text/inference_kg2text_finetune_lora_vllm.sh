#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BasePath=/data/home/usera401
BaseMODEL=${BasePath}/data/pretrained-models/llama2-7b-chat
LoRAMODEL=$1

#DATA=${BasePath}/data/AMRData/LDC2020-ori-amr/test.jsonl
DATA=${BasePath}/data/KGData/webnlg/test.jsonl
DATA=${BasePath}/data/TaskData/webnlgv2.1/test.jsonl

PORT_ID=$(expr $RANDOM + 1000)

python -u ../inference_vllm.py \
    --test_file ${DATA} \
    --lora_name_or_path ${LoRAMODEL} \
    --model_name_or_path ${BaseMODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_prefix "LoRA-7b-chat-pred" \
    --input_key "input" \
    --instruction "" \
    --prompt_template "llama2-chat-kg" 2>&1 | tee ${LoRAMODEL}/eval.log