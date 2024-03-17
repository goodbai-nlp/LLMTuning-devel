#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
export PATH="/root/micromamba/bin:$PATH"
source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

BasePath=/baixuefeng
PORT_ID=$(expr $RANDOM + 1000)
DataPath=${BasePath}/data/ConData

export HF_DATASETS_CACHE=${BasePath}/data/.cache
MODEL=$1
DataSetName=1219-wsj-pycode
DATA=${DataPath}/${DataSetName}/test.jsonl
# DATA=${DataPath}/${DataSetName}/debug.jsonl

python -u inference_vllm.py --test_file ${DATA} --model_name_or_path ${MODEL} --max_new_tokens 1536 --out_prefix "test-pred-${DataSetName}" --prompt_template "supervised-conparsing-pycode" 2>&1 | tee $MODEL/eval.log

python -u convert_tree_to_lisp.py $MODEL/test-pred-${DataSetName}_test.jsonl_pred_fast $MODEL/test-pred-${DataSetName}_test.jsonl_pred_fast_tree
cd EVALB
python evaluate_bra_ori.py $MODEL/test-pred-${DataSetName}_test.jsonl_pred_fast_tree /baixuefeng/data/ConData/0525-wsj-test/gold.txt 2>&1 | tee $MODEL/eval-${DataSetName}.txt
# python evaluate_bra_ori.py $MODEL/test-pred-${DataSetName}_debug.jsonl_pred_fast /baixuefeng/data/ConData/0525-wsj-test/gold-debug.txt 2>&1 | tee $MODEL/eval-${DataSetName}.txt