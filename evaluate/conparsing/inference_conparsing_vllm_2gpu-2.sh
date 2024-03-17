#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
source /etc/profile.d/modules.sh 
export MODULEPATH=/ssd/apps/apps/modulefiles:$MODULEPATH
module load anaconda/2021.11
module load cuda/11.7
source activate LLMTuning
#export RAY_ADDRESS="localhost:80886"

for cate in 0.0
do
MODEL=/data/model/llama-65b/
DATA=/data/xfbai/data/TaskData/ConData/test-zero-shot.jsonl
python -u inference_vllm.py --test_file ${DATA} --model_name_or_path ${MODEL} --num_beams 1 --frequency_penalty $cate --max_new_tokens 512 --out_prefix "test-pred-${cate}-zeroshot" --input_key "sentence" --instruction "" --prompt_template "None" 2>&1 | tee $MODEL/eval.log
done
