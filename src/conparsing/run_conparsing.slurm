#!/bin/bash
#SBATCH --job-name=xfbai-Conparsing     #作业名称
#SBATCH --partition=q_intel_share       #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=96            #申请每个节点32核CPU
#SBATCH --gres=gpu:8                    #申请4张GPU卡
#SBATCH -w wxhd11                           #指定GPU节点
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load Anaconda cuda-11.8 gcc-9.3.0
source activate py3.10torch2.0devel
cd $SLURM_SUBMIT_DIR
#bash finetune_conparsing_clm_70B.sh
bash finetune_conparsing_clm_13B.sh
