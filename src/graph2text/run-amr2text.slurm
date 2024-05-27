#!/usr/bin/bash
#SBATCH --job-name=xfbai-RLHF     #作业名称
#SBATCH --partition=hitsz_mzhang       #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=8            #申请每个节点32核CPU
#SBATCH --nodelist=gn[06]
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load anaconda/2023.03 cuda/cuda-11.8 nccl/2.18.1-cuda-11.8 pdsh/2.31-gcc-9.4.0
source activate py3.10torch2.1devel
cd $SLURM_SUBMIT_DIR
bash finetune_amr2text_clm.sh
