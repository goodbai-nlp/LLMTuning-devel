#!/bin/bash
#SBATCH --job-name=xfbai-Conparsing     #作业名称
#SBATCH --partition=q_intel_share       #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=64            #申请每个节点64核CPU
#SBATCH --gres=gpu:4                    #申请4张GPU卡
#SBATCH -w wxhd10                       #（必要内容）指定运行节点，不指定无法确保找到对应的数据
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load Anaconda cuda-11.7 gcc-9.3.0
source activate py3.10torch2.0
cd $SLURM_SUBMIT_DIR
echo ${SLURM_SUBMIT_DIR}
bash xxx.sh				#执行用户程序
