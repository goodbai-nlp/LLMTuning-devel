#!/bin/bash
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $ROOT_DIR

MODEL=$1
DATA=/data/data/TaskData/LDC2020-amr/test.jsonl
BaseModel=/data/data/pretrained-models/Llama-2-7b-hf
#python -u inference_peft.py --test_file ${DATA} --base_model_name_or_path ${BaseModel} --model_name_or_path ${MODEL} --num_beams 5 --max_new_tokens 512 --out_prefix "test-pred" --input_key "amr" --instruction "Generate a descriptive text for the given abstract meaning representation graph." --prompt_template "supervised" 2>&1 | tee $MODEL/eval.log
python -u inference_peft.py --test_file ${DATA} --base_model_name_or_path ${BaseModel} --model_name_or_path ${MODEL} --num_beams 1 --max_new_tokens 400 --out_prefix "test-pred" --input_key "amr" --instruction "Generate a descriptive text for the given abstract meaning representation graph." --prompt_template "supervised" 2>&1 | tee $MODEL/eval.log
python eval_bleu.py ${DATA} ${MODEL}/test-pred_test.jsonl_pred_peft
