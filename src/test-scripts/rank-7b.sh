#!/bin/bash
source /etc/profile.d/modules.sh 
export MODULEPATH=/ssd/apps/apps/modulefiles:$MODULEPATH
module load anaconda/2021.11
module load cuda/11.7
source activate LLMTuning
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23

# nodes
NODES=$1

# gpus
NUM_GPUS=$2

# rank
NODE_RANK=$3

# Master addr
MASTER_ADDR=$4
MASTER_PORT=29501

#DHOSTFILE
DHOSTFILE=$5

# JOB_ID
JOB_ID=$6

# world_info
WORLD_INFO=`python gen_world_info.py $NODES $NUM_GPUS $DHOSTFILE | tail -1`

# logs
OUTPUT_LOG="${JOB_ID}/train_rank${NODE_RANK}.log"
echo "nodes,gpus,mp_size,node_rank,master_addr,master_port,dhostfile" >> $OUTPUT_LOG
echo "$NODES,$NUM_GPUS,$MP_SIZE,$NODE_RANK,$MASTER_ADDR,$MASTER_PORT,$DHOSTFILE,$WORLD_INFO" >> $OUTPUT_LOG

BasePath=/data/xfbai
MODEL=${BasePath}/data/pretrained-models/llama-7b
MODEL=/data/model/llama-65b

DataPath=${BasePath}/data/TaskData
DataSetName=stanford-alpaca
DataSetName=pubmedqa
DataSetName=trucated-pubmedqa

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

lr=2e-5

OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz128-decay0.1-3epoch
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz128-decay0.1-3epoch-NewToken-V2
# OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-65b-GenMode-lr-${lr}-totalbsz32-nooffload
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-65b-GenMode-lr-${lr}-totalbsz32-stage2-offload
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-65b-GenMode-lr-${lr}-totalbsz128-stage3-offload
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-65b-GenMode-lr-${lr}-totalbsz256-stage3-offload-2Nodes
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz256-stage3-offload-2Nodes
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz256-stage2-nooffload-GC-2Nodes
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz512-stage3-offload-GC-2Nodes
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-65b-GenMode-lr-${lr}-totalbsz32-stage3-nooffload

rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR}

#if [ ! -d ${OUTPUT_DIR} ];then
#  mkdir -p ${OUTPUT_DIR}
#else
#  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
#  case $yn in
#    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
#    [Nn]* ) echo "exiting..."; exit;;
#    * ) echo "Please answer yes or no.";;
#  esac
#fi

MODEL_SIZE=7B
# NUM_GPUS=8
# NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

#deepspeed --hostfile host_file benchmark.py 
which python
python -u -m deepspeed.launcher.launch \
     --world_info=${WORLD_INFO} \
     --node_rank=${NODE_RANK} \
     --master_addr=${MASTER_ADDR} \
     --master_port=${MASTER_PORT} \
     benchmark.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --conditional_gen False \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
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
    --logging_first_step True \
    --gradient_checkpointing \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_output_dir \
    --preprocessing_num_workers 1 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache \
    --report_to "tensorboard" >> $OUTPUT_LOG 2>&1

