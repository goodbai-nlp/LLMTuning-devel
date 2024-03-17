#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export PATH="/opt/conda/bin:$PATH"
#source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

BasePath=/baixuefeng
DATA=${BasePath}/data/AMRData/LDC2020-mtl-raw/test-NodeEdge.jsonl
DATA=${BasePath}/data/AMRData/LDC2020-mtl/test-Sent.jsonl
DATA=${BasePath}/data/AMRData/LDC2020-amr2text/test.jsonl
DATA=${BasePath}/data/AMRData/LDC2020-amr/test.jsonl
#DATA=${BasePath}/data/AMRData/LDC2020-ori-amr/test.jsonl

PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
python -u inference_hf.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 512 \
    --out_prefix "pred" \
    --input_key "amr" \
    --instruction "Generate a descriptive text for the given graph." \
    --prompt_template "supervised" 2>&1 | tee $MODEL/eval.log

echo "Evaluating SacreBLEU score ..."
python eval_bleu.py ${DATA} output/pred_test_hf

gold=${BasePath}/data/AMRData/LDC2020-amr2text/test-gold.txt
gold_tok=${gold}.tok
pred=output/pred_test_hf
pred_tok=${pred}.tok

tokenizer=cdec-corpus/tokenize-anything.sh
echo "Tokenizing files ..."
bash $tokenizer -u  < $gold > $gold_tok
bash $tokenizer -u  < $pred > $pred_tok
echo "Evaluating tokenized BLEU score ..."
python eval_gen.py --in-tokens $pred_tok --in-reference-tokens $gold_tok
