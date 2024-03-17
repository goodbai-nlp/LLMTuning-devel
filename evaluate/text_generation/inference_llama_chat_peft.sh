#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0

BaseMODEL=/data/home/usera401/data/pretrained-models/llama2-7b-chat
BaseMODEL=/data/home/usera401/data/pretrained-models/llama2-13b-chat
LoRAMODEL=$1

DATA=./test-data/test.json
DATA=./test-data/test_0830.json
DATA=./test-data/test-0219.json
DATA=./test-data/test-random.json
#DATA=./test-data/test-domain.json
DATA=/data/home/usera401/data/TaskData/task7_split/test.json

python -u ../inference_vllm.py --test_file ${DATA} --model_name_or_path ${BaseMODEL} --lora_name_or_path ${LoRAMODEL} --num_beams 1 --max_new_tokens 512 --out_prefix "LoRA-7b-chat-pred" --instruction "" --input_key "Question" --prompt_template "llama2-chat" 2>&1 | tee $LoRAMODEL/eval.log
