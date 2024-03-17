#!/bin/bash
#SBATCH --job-name=xfbai-Graph2text     #作业名称
#SBATCH --partition=q_intel_share       #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=16            #申请每个节点32核CPU
#SBATCH --gres=gpu:1                    #申请4张GPU卡
#SBATCH -w wxhd09           		       #指定GPU
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load Anaconda cuda-11.7 gcc-9.3.0
source activate py3.10torch2.0
cd $SLURM_SUBMIT_DIR
#bash inference_amr2text_finetune_vllm_v2.sh /cpfs/29bcf0bdae829206-000001/home/usera401/output/exp.LLMTuning/Finetune-LDC2020-leo-amr2text-llama2-7b-ConditionalGenMode-lr-2e-5-totalbsz128-decay0.1-5epoch-bsz128-Newest/checkpoint-435
bash inference_amr2text_finetune_vllm.sh /cpfs/29bcf0bdae829206-000001/home/usera401/output/exp.LLMTuning/Finetune-LDC2020-var-amr2text-llama2-7b-ConditionalGenMode-lr-2e-5-totalbsz128-decay0.1-5epoch-bsz128-Newest-Reproduce/checkpoint-last
