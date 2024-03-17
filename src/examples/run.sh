BasePath=/share/hpc/home/usersa/usera409
CUDA_VISIBLE_DEVICES=0 python run_clm.py \
    --model_name_or_path ${BasePath}/data/pretrained-models/gpt2-large \
    --dataset_name ${BasePath}/data/wikitext \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --overwrite_output_dir \
    --output_dir test-clm
