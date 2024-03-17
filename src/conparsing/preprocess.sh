export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# BasePath=/cpfs/29bcf0bdae829206-000001/home/usera401/
BasePath=/data/home/usera401
CurDir=$(cd $(dirname $0);cd ..; pwd)


MODEL_NAME=llama2-7b

MODEL=${BasePath}/data/pretrained-models/${MODEL_NAME}
DataPath=${BasePath}/data/ConData/
DataSetName=0525-wsj
DataSetName=raw-wsj

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

lr=3e-5
NUM_EPOCHS=5

OUTPUT_DIR=${BasePath}/output/exp.LLMTuning/preprocess-${DataSetName}-${MODEL_NAME}-ConditionalGenMode

if [ ! -d ${OUTPUT_DIR} ];then
  mkdir -p ${OUTPUT_DIR}
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

echo ${CurDir}
python -u ${CurDir}/finetune_std.py \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --conditional_gen True \
    --seed 42 \
    --max_seq_length 1024 \
    --do_preprocess \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --evaluation_strategy "steps" \
    --logging_steps 100 \
    --greater_is_better False \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --ignore_opt_states True \
    --num_train_epochs ${NUM_EPOCHS} \
    --logging_first_step True \
    --gradient_checkpointing \
    --use_peft False \
    --use_flash_attn True \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 1 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache-clm \
    --report_to "wandb" 2>&1 | tee ${OUTPUT_DIR}/training.log