#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BasePath=/UNICOMFS/hitsz_khchen_1

DATA=${BasePath}/data/KGData/webnlg/test.jsonl
datacate=webnlg17
datacate=webnlg20
datacate=EventNarrative

DATA=${BasePath}/data/Data2text/${datacate}/test.jsonl
#DATA=${BasePath}/data/Data2text/${datacate}new/test.jsonl

PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
python -u ../inference_vllm.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_prefix "Finetuned-7b-pred" \
    --input_key "src" \
    --instruction "Generate a coherent piece of text that contains all of the information in the triples." \
    --prompt_template "vicuna" 2>&1 | tee $MODEL/eval.log

# python eval_webnlg.py $MODEL/Finetuned-7b-pred_test_vllm.txt ${datacate}
