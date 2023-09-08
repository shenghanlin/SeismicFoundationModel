CUDA_VISIBLE_DEVICES='5' OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
    --data_path '../Data/Denoise/' \
    --task 'Denoise' \
    --accum_iter 1 \
    --batch_size 60 \
    --input_size 224 \
    --model vit_base_patch16 \
    --output_dir './finetune_result/Denoise/net_Base_scratch/' \
    --log_dir './finetune_result/Denoise/net_Base_scratch/' \
    --epochs 100 \
    --warmup_epochs 10 \
    --lr 1.5e-3 --weight_decay 0.05 \
    --layer_decay 0.05 --drop_path 0.1 --reprob 0.25 \
    --dist_eval \
    
