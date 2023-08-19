#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/gpfs/home/ess/hlsheng/.local/lib/python3.9/site-packages
python submitit_pretrain.py \
    --job_dir '/gpfs/home/ess/hlsheng/mae-main/output_base_gpu4/' \
    --batch_size 580\
    --accum_iter 4 \
    --model mae_vit_base_patch16D4d256 \
    --resume './output_base_gpu4/checkpoint-1516.pth' \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path '../mae_data_more/'

