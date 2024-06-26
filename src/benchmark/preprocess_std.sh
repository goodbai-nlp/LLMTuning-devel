export CUDA_VISIBLE_DEVICES=0

BasePath=/home/export/base/ycsc_chenkh/chenkh_nvlink/online1/xfbai
MODEL=${BasePath}/data/pretrained-models/llama2-7b

DataPath=${BasePath}/data/TaskData
DataSetName=trucated-pubmedqa

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

lr=2e-5
CurDir=$(cd $(dirname $0);cd ..; pwd)
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama2-7b-lr-${lr}-totalbsz128-decay0.1-3epoch

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

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed ${CurDir}/finetune_std.py \
    --deepspeed ${CurDir}/ds_configs/stage2_no_offloading.conf \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --conditional_gen False \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
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
    --save_total_limit 3 \
    --ignore_opt_states True \
    --num_train_epochs 3 \
    --gradient_checkpointing \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 1 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log
